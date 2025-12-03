import math
from typing import Optional, Dict

import torch
import torch.nn.functional as F

from .registery import AdaptiveModel, register
from . import functional


def _update_ema(prev: Optional[torch.Tensor],
                value: torch.Tensor,
                alpha: float) -> torch.Tensor:
    """Exponential moving average helper."""
    if prev is None:
        return value.detach()
    return alpha * prev + (1.0 - alpha) * value.detach()


@torch.no_grad()
def _interp_state_dict(current: Dict[str, torch.Tensor],
                       init_state: Dict[str, torch.Tensor],
                       lam: float) -> Dict[str, torch.Tensor]:
    """
    Soft reset interpolation:
        theta_new = lam * theta_init + (1 - lam) * theta_current
    """
    new_state = {}
    for k, v_cur in current.items():
        v_init = init_state.get(k, None)
        if v_init is None:
            new_state[k] = v_cur
        else:
            if (torch.is_floating_point(v_cur)
                and torch.is_floating_point(v_init)
                and v_cur.shape == v_init.shape):
                new_state[k] = lam * v_init + (1.0 - lam) * v_cur
            else:
                new_state[k] = v_cur
    return new_state


class _RDumbPPBase(AdaptiveModel):
    """
    RDumb++ base class.

    Extends RDumb with:
      - Entropy-based drift detection  OR
      - KL-based drift detection

    and:
      - full reset (load original RDumb checkpoint) OR
      - soft reset (interpolate to original weights).
    """

    def __init__(
        self,
        model,
        entropy_ema_alpha: float = 0.99,
        drift_k: float = 0.25,
        kl_ema_alpha: float = 0.99,
        soft_lambda: float = 0.5,
        warmup_steps: int = 5,
        cooldown_steps: int = 20,
    ):
        super().__init__(model)

        # === RDumb parameters (same as original) ===
        # Entropy margin used for reliability filtering
        self.e_margin = math.log(1000) * 0.4  # keep as in RDumb
        self.d_margin = 0.05
        self.current_model_probs = None  # used for redundancy filtering

        # Prepare model + optimizer as in RDumb
        params, _ = functional.collect_params(model)
        model = functional.configure_model(model)

        self.model = model
        self.optimizer = torch.optim.SGD(params, lr=0.00025, momentum=0.9)

        # RDumb checkpoint (for full reset)
        self.model_state, self.optimizer_state = functional.copy_model_and_optimizer(
            self.model, self.optimizer
        )
        # Soft reset checkpoint (same initial state)
        self.init_model_state, _ = functional.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # === Drift statistics ===
        # Entropy EMA
        self.entropy_ema: Optional[torch.Tensor] = None

        # KL EMA + reference distribution
        self.kl_ema: Optional[torch.Tensor] = None
        self.kl_ref_probs: Optional[torch.Tensor] = None

        # Hyperparameters
        self.entropy_ema_alpha = entropy_ema_alpha
        self.kl_ema_alpha = kl_ema_alpha
        self.soft_lambda = soft_lambda
        self.drift_k = drift_k          # threshold on |current - EMA|
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps

        # Step tracking
        self.total_steps = 0
        self.steps_since_reset = 0

        # To be set by subclasses
        self.drift_mode = None   # "entropy" | "kl"
        self.reset_mode = None   # "full" | "soft"

    # =====================
    #     DRIFT DETECTION
    # =====================

    def _entropy_drift(self, outputs: torch.Tensor) -> float:
        """
        Drift based on absolute change in batch-mean entropy:
            drift = | H_t - EMA(H) |
        drift_k is interpreted in "entropy units" (≈ 0.1–0.5).
        """
        batch_ent = functional.softmax_entropy(outputs).mean()

        if self.entropy_ema is None:
            # First batch: initialize, no drift yet
            self.entropy_ema = batch_ent.detach()
            return 0.0

        # Drift = |current - EMA|
        drift = (batch_ent - self.entropy_ema).abs().item()

        # Update EMA
        self.entropy_ema = _update_ema(self.entropy_ema, batch_ent, self.entropy_ema_alpha)

        return drift

    def _kl_drift(self, outputs: torch.Tensor) -> float:
        """
        Drift based on absolute change in KL(mean_probs || EMA_ref_probs):
            drift = | KL_t - EMA(KL) |   (implicitly via EMA of KL)
        drift_k is in "KL units" (≈ 0.005–0.05).
        """
        probs = outputs.softmax(dim=1).detach()
        mean_probs = probs.mean(dim=0)

        if self.kl_ref_probs is None:
            # First batch: set reference distribution
            self.kl_ref_probs = mean_probs.detach()
            self.kl_ema = torch.tensor(0.0, device=mean_probs.device)
            return 0.0

        ref = self.kl_ref_probs.clamp(min=1e-6)
        cur = mean_probs.clamp(min=1e-6)
        kl_val = F.kl_div(cur.log(), ref, reduction="batchmean")

        if self.kl_ema is None:
            self.kl_ema = kl_val.detach()
            drift = 0.0
        else:
            drift = (kl_val - self.kl_ema).abs().item()

        # Update EMA of KL
        self.kl_ema = _update_ema(self.kl_ema, kl_val, self.kl_ema_alpha)
        # Slowly adapt reference distribution
        self.kl_ref_probs = 0.9 * self.kl_ref_probs + 0.1 * mean_probs.detach()

        return drift

    # =====================
    #     RESET DECISION
    # =====================

    def _should_reset(self, outputs: torch.Tensor) -> bool:
        """
        Decide whether to trigger a reset based on:
          - minimum warmup,
          - cooldown since last reset,
          - drift magnitude.
        """
        if self.total_steps < self.warmup_steps:
            return False
        if self.steps_since_reset < self.cooldown_steps:
            return False

        if self.drift_mode == "entropy":
            drift = self._entropy_drift(outputs)
        elif self.drift_mode == "kl":
            drift = self._kl_drift(outputs)
        else:
            return False

        # Optional debug:
        # if self.total_steps % 200 == 0:
        #     print(f"[RDumb++] step={self.total_steps}, drift={drift:.4f}, mode={self.drift_mode}")

        return drift > self.drift_k

    # =====================
    #      APPLY RESET
    # =====================

    def _apply_reset(self):
        """
        Apply full or soft reset.

        NOTE:
        We DO NOT clear EMA stats here, so drift detection remains meaningful
        after the reset. We only reset adaptation-related state.
        """
        if self.reset_mode == "full":
            # Restore original RDumb checkpoint
            functional.load_model_and_optimizer(
                self.model, self.optimizer, self.model_state, self.optimizer_state
            )

        elif self.reset_mode == "soft":
            with torch.no_grad():
                current_state = self.model.state_dict()
                new_state = _interp_state_dict(
                    current_state, self.init_model_state, self.soft_lambda
                )
                self.model.load_state_dict(new_state)

            # Refresh optimizer state after soft reset
            _, opt_state = functional.copy_model_and_optimizer(self.model, self.optimizer)
            self.optimizer.load_state_dict(opt_state)

        # Reset RDumb "memory" of model probs
        self.current_model_probs = None

        # Do NOT clear EMAs here → lets drift accumulate again
        # self.entropy_ema, self.kl_ema, self.kl_ref_probs stay as is.

        self.steps_since_reset = 0

    # =====================
    #        FORWARD
    # =====================

    @torch.enable_grad()
    def forward(self, x):
        """
        Forward pass with:
          1) possible dynamic reset,
          2) RDumb-style adaptation update.
        """
        # Forward (pre-adaptation)
        outputs = self.model(x)

        # Check for drift-triggered reset
        if self._should_reset(outputs):
            self._apply_reset()
            # Recompute outputs after reset
            outputs = self.model(x)

        # ===== RDumb-style adaptation =====
        entropys = functional.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ent_f1 = entropys[filter_ids_1]

        outputs_softmax = outputs.softmax(dim=1)

        if self.current_model_probs is not None:
            sim = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0),
                outputs_softmax[filter_ids_1].detach(),
                dim=1,
            )
            filter_ids_2 = torch.where(sim.abs() < self.d_margin)
            ent_f2 = ent_f1[filter_ids_2]
            new_probs = outputs_softmax[filter_ids_1][filter_ids_2].detach()
        else:
            # No redundancy filtering yet
            filter_ids_2 = torch.where(filter_ids_1[0] > -1)
            ent_f2 = ent_f1
            new_probs = outputs_softmax[filter_ids_1].detach()

        # Update RDumb EMA of model probabilities ⟨p⟩
        if new_probs.size(0) > 0:
            if self.current_model_probs is None:
                self.current_model_probs = new_probs.mean(0)
            else:
                self.current_model_probs = (
                    0.9 * self.current_model_probs + 0.1 * new_probs.mean(0)
                )

        # RDumb entropy reweighting
        if ent_f2.numel() > 0:
            coeff = 1.0 / torch.exp(ent_f2 - self.e_margin)
            loss = (ent_f2 * coeff).mean()

            # Gradient update
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            # No reliable samples in this batch
            self.optimizer.zero_grad(set_to_none=True)

        self.total_steps += 1
        self.steps_since_reset += 1

        return outputs


# ============================
#       MODEL VARIANTS
# ============================

@register("rdumbpp_ent_full")
class RDumbPPEntropyFull(_RDumbPPBase):
    """
    RDumb++ with entropy-based drift detection and FULL resets.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "entropy"
        self.reset_mode = "full"


@register("rdumbpp_ent_soft")
class RDumbPPEntropySoft(_RDumbPPBase):
    """
    RDumb++ with entropy-based drift detection and SOFT resets.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "entropy"
        self.reset_mode = "soft"


@register("rdumbpp_kl_full")
class RDumbPPKLFull(_RDumbPPBase):
    """
    RDumb++ with KL-based drift detection and FULL resets.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "kl"
        self.reset_mode = "full"


@register("rdumbpp_kl_soft")
class RDumbPPKLSoft(_RDumbPPBase):
    """
    RDumb++ with KL-based drift detection and SOFT resets.
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "kl"
        self.reset_mode = "soft"

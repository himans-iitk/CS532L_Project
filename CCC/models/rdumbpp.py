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
    return alpha * prev + (1 - alpha) * value.detach()


@torch.no_grad()
def _interp_state_dict(current: Dict[str, torch.Tensor],
                       init_state: Dict[str, torch.Tensor],
                       lam: float) -> Dict[str, torch.Tensor]:
    """Soft reset interpolation."""
    new_state = {}
    for k, v_cur in current.items():
        v_init = init_state.get(k, None)
        if v_init is None:
            new_state[k] = v_cur
        else:
            if (torch.is_floating_point(v_cur) and
                torch.is_floating_point(v_init) and
                v_cur.shape == v_init.shape):
                new_state[k] = lam * v_init + (1 - lam) * v_cur
            else:
                new_state[k] = v_cur
    return new_state


class _RDumbPPBase(AdaptiveModel):
    """
    RDumb++ base class:
    Supports:
      - entropy drift detection
      - KL drift detection
      - full reset
      - soft reset
    """
    def __init__(
        self, model,
        entropy_ema_alpha=0.99,
        drift_k=3.0,
        kl_ema_alpha=0.99,
        soft_lambda=0.5,
        warmup_steps=50,
        cooldown_steps=200
    ):
        super().__init__(model)

        # RDumb parameters
        self.e_margin = math.log(1000) * 0.4
        self.d_margin = 0.05
        self.current_model_probs = None

        params, _ = functional.collect_params(model)
        model = functional.configure_model(model)

        self.model = model
        self.optimizer = torch.optim.SGD(params, lr=0.00025, momentum=0.9)

        # RDumb checkpoint
        self.model_state, self.optimizer_state = functional.copy_model_and_optimizer(
            self.model, self.optimizer
        )
        # soft reset checkpoint
        self.init_model_state, _ = functional.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # drift stats
        self.entropy_ema = None
        self.entropy_ema_sq = None
        self.kl_ema = None
        self.kl_ema_sq = None

        # hyperparameters (EXTERNALLY TUNED)
        self.entropy_ema_alpha = entropy_ema_alpha
        self.kl_ema_alpha = kl_ema_alpha
        self.soft_lambda = soft_lambda
        self.drift_k = drift_k
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps

        # step tracking
        self.total_steps = 0
        self.steps_since_reset = 0

        # overridden by subclasses
        self.drift_mode = None
        self.reset_mode = None


    # ===== Drift detection =====
    def _entropy_drift(self, outputs):
        batch_ent = functional.softmax_entropy(outputs).mean()
        self.entropy_ema = _update_ema(self.entropy_ema, batch_ent, self.entropy_ema_alpha)
        self.entropy_ema_sq = _update_ema(self.entropy_ema_sq, batch_ent * batch_ent, self.entropy_ema_alpha)

        var = torch.clamp(self.entropy_ema_sq - self.entropy_ema**2, min=1e-6)
        std = torch.sqrt(var)
        z = ((batch_ent - self.entropy_ema) / std).item()
        return z

    def _kl_drift(self, outputs):
        probs = outputs.softmax(dim=1).detach()
        mean_probs = probs.mean(dim=0)

        if self.current_model_probs is None:
            self.current_model_probs = mean_probs.detach()
            return 0.0

        # update EMA
        self.current_model_probs = 0.9 * self.current_model_probs + 0.1 * mean_probs.detach()

        ref = self.current_model_probs.clamp(min=1e-6)
        cur = mean_probs.clamp(min=1e-6)
        kl_val = F.kl_div(cur.log(), ref, reduction="batchmean")

        self.kl_ema = _update_ema(self.kl_ema, kl_val, self.kl_ema_alpha)
        self.kl_ema_sq = _update_ema(self.kl_ema_sq, kl_val*kl_val, self.kl_ema_alpha)

        var = torch.clamp(self.kl_ema_sq - self.kl_ema**2, min=1e-8)
        std = torch.sqrt(var)
        z = ((kl_val - self.kl_ema) / std).item()
        return z


    # ===== Reset decision =====
    def _should_reset(self, outputs):
        if self.total_steps < self.warmup_steps:
            return False
        if self.steps_since_reset < self.cooldown_steps:
            return False

        if self.drift_mode == "entropy":
            z = self._entropy_drift(outputs)
        elif self.drift_mode == "kl":
            z = self._kl_drift(outputs)
        else:
            return False

        return z > self.drift_k


    # ===== Apply reset (full / soft) =====
    def _apply_reset(self):
        if self.reset_mode == "full":
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

            # refresh optimizer
            _, opt_state = functional.copy_model_and_optimizer(self.model, self.optimizer)
            self.optimizer.load_state_dict(opt_state)

        # reset drift stats
        self.entropy_ema = None
        self.entropy_ema_sq = None
        self.kl_ema = None
        self.kl_ema_sq = None
        self.current_model_probs = None
        self.steps_since_reset = 0


    # ===== Forward =====
    @torch.enable_grad()
    def forward(self, x):
        outputs = self.model(x)

        # dynamic reset
        if self._should_reset(outputs):
            self._apply_reset()
            outputs = self.model(x)

        # ===== RDumb-style adaptation =====
        entropys = functional.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ent_f1 = entropys[filter_ids_1]

        outputs_softmax = outputs.softmax(dim=1)
        if self.current_model_probs is not None:
            sim = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0),
                outputs_softmax[filter_ids_1].detach(), dim=1
            )
            filter_ids_2 = torch.where(sim.abs() < self.d_margin)
            ent_f2 = ent_f1[filter_ids_2]
            new_probs = outputs_softmax[filter_ids_1][filter_ids_2].detach()
        else:
            filter_ids_2 = torch.where(filter_ids_1[0] > -1)
            ent_f2 = ent_f1
            new_probs = outputs_softmax[filter_ids_1].detach()

        # update EMA of model probs ⟨p⟩
        if new_probs.size(0) > 0:
            if self.current_model_probs is None:
                self.current_model_probs = new_probs.mean(0)
            else:
                self.current_model_probs = 0.9 * self.current_model_probs + 0.1 * new_probs.mean(0)

        # RDumb entropy weighting
        coeff = 1 / torch.exp(ent_f2 - self.e_margin)
        loss = (ent_f2 * coeff).mean()

        # gradient update
        if new_probs.size(0) > 0:
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        self.total_steps += 1
        self.steps_since_reset += 1

        return outputs


@register("rdumbpp_ent_full")
class RDumbPPEntropyFull(_RDumbPPBase):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "entropy"
        self.reset_mode = "full"


@register("rdumbpp_ent_soft")
class RDumbPPEntropySoft(_RDumbPPBase):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "entropy"
        self.reset_mode = "soft"


@register("rdumbpp_kl_full")
class RDumbPPKLFull(_RDumbPPBase):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "kl"
        self.reset_mode = "full"


@register("rdumbpp_kl_soft")
class RDumbPPKLSoft(_RDumbPPBase):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.drift_mode = "kl"
        self.reset_mode = "soft"

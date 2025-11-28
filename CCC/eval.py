import argparse
import os
import torch

# Import torchvision with error handling for CUDA version mismatches
try:
    import torchvision.models as models
    import torchvision.transforms as trn
except RuntimeError as e:
    if "different CUDA versions" in str(e) or "CUDA Version" in str(e):
        pytorch_cuda = torch.version.cuda if torch.cuda.is_available() else None
        print(f"\n✗ ERROR: CUDA version mismatch detected!")
        print(f"  PyTorch CUDA version: {pytorch_cuda}")
        print(f"  Error: {e}")
        raise RuntimeError("CUDA and torchvision versions do not match.")
    else:
        raise

import webdataset as wds
from models import registery


def identity(x):
    return x


def get_webds_loader(dset_name, dset_path=None):
    """Load CCC dataset using WebDataset."""
    if dset_path:
        url = os.path.join(dset_path, dset_name, "serial_{{00000..99999}}.tar")
    else:
        url = f'https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/{dset_name}/serial_{{00000..99999}}.tar'

    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    preproc = trn.Compose([trn.ToTensor(), normalize])

    dataset = (
        wds.WebDataset(url)
        .decode("pil")
        .to_tuple("input.jpg", "output.cls")
        .map_tuple(preproc, identity)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)
    return loader


def test(model, dset_path, file_name=None, local_dset_path=None):
    """Evaluate accuracy along the CCC stream."""
    total_seen_so_far = 0
    dataset_loader = get_webds_loader(dset_path, local_dset_path)

    device = next(model.parameters()).device

    for i, (images, labels) in enumerate(dataset_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        preds = output.argmax(dim=1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum().item()

        with open(file_name, "a+") as f:
            f.write(f"acc_{100 * correct / images.size(0):.10f}\n")

        total_seen_so_far += images.size(0)
        if total_seen_so_far > 7_500_000:
            return


def evaluate(args):
    torch.manual_seed(42)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    if cuda_available:
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available. Running on CPU (slow).")

    baseline_int = int(args.baseline)
    exp_name = f"ccc_{baseline_int}"

    os.makedirs(os.path.join(args.logs, exp_name), exist_ok=True)

    cur_seed = [43, 44, 45][args.processind % 3]
    speed = [1000, 2000, 5000][args.processind // 3]

    file_name = os.path.join(
        args.logs, exp_name,
        f"model_{args.mode}_baseline_{baseline_int}_transition+speed_{speed}_seed_{cur_seed}.txt"
    )

    dset_name = f"baseline_{baseline_int}_transition+speed_{speed}_seed_{cur_seed}"

    # Load pretrained backbone
    model = models.resnet50(pretrained=True).to(device)

    # Multi-GPU support (optional)
    if cuda_available and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    assert args.mode in registery.get_options(), \
           f"Unknown mode {args.mode}. Available: {registery.get_options()}"

    # Special case: ETA/EATA needs loader
    if args.mode in ["eta", "eata"]:
        loader = get_webds_loader(dset_name, args.dset)
        model = registery.init(args.mode, model, loader, args.mode == "eta")
    elif args.mode.startswith("rdumbpp_"):
        # RDumb++ models need these kwargs
        model = registery.init(
            args.mode, model,
            drift_k=args.drift_k,
            warmup_steps=args.warmup,
            cooldown_steps=args.cooldown,
            soft_lambda=args.lambda_soft,
            entropy_ema_alpha=args.ent_alpha,
            kl_ema_alpha=args.kl_alpha
        )
    else:
        # Tent, RDumb, Pretrained, etc. - only need the model
        model = registery.init(args.mode, model)

    test(model, dset_name, file_name=file_name,
         local_dset_path=args.dset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="tent",
                        choices=registery.get_options())
    parser.add_argument("--processind", type=int, default=0)
    parser.add_argument("--baseline", type=float, default=20)
    parser.add_argument("--logs", type=str, required=True)
    parser.add_argument("--dset", type=str, default=None)

    # ---- RDumb++ PARAMETERS ----
    parser.add_argument("--drift_k", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--cooldown", type=int, default=200)
    parser.add_argument("--lambda_soft", type=float, default=0.5)
    parser.add_argument("--ent_alpha", type=float, default=0.99)
    parser.add_argument("--kl_alpha", type=float, default=0.99)

    args = parser.parse_args()
    evaluate(args)

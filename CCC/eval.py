import argparse
import os

import torch
import torchvision.models as models
import torchvision.transforms as trn
import webdataset as wds

from models import registery


def identity(x):
    return x


def get_webds_loader(dset_name, dset_path=None):
    # Use local dataset if path is provided, otherwise stream from cloud
    if dset_path:
        url = os.path.join(dset_path, dset_name, "serial_{{00000..99999}}.tar")
    else:
        url = f'https://mlcloud.uni-tuebingen.de:7443/datasets/CCC/{dset_name}/serial_{{00000..99999}}.tar'

    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preproc = trn.Compose(
        [
            trn.ToTensor(),
            normalize,
        ]
    )
    dataset = (
        wds.WebDataset(url)
        .decode("pil")
        .to_tuple("input.jpg", "output.cls")
        .map_tuple(preproc, identity)
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=64)
    return dataloader


def test(model, dset_path, file_name=None, local_dset_path=None):
    total_seen_so_far = 0
    dataset_loader = get_webds_loader(dset_path, local_dset_path)
    
    # Get device from model
    device = next(model.parameters()).device

    for i, (images, labels) in enumerate(dataset_loader):
        # Use device-agnostic transfer
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        num_images_in_batch = images.size(0)
        total_seen_so_far += num_images_in_batch

        vals, pred = (output).max(dim=1, keepdim=True)
        correct_this_batch = pred.eq(labels.view_as(pred)).sum().item()

        with open(file_name, "a+") as f:
            f.write(
                ("acc_{:.10f}\n").format(
                    float(100 * correct_this_batch) / images.size(0)
                )
            )
        if total_seen_so_far > 7500000:
            return


def evaluate(args):
    torch.manual_seed(42)
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise ValueError("CUDA not available. GPU is required for evaluation. "
                        "Please ensure PyTorch is installed with CUDA support and GPU is accessible.")
    device = torch.device("cuda")
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    # Convert baseline to int for directory name
    baseline_int = int(args.baseline)
    exp_name = "ccc_{}".format(baseline_int)

    if not os.path.exists(os.path.join(args.logs, exp_name)):
        os.mkdir(os.path.join(args.logs, exp_name))

    cur_seed = [43, 44, 45][args.processind % 3]
    speed = [1000, 2000, 5000][int(args.processind / 3)]

    file_name = os.path.join(
        args.logs,
        exp_name,
        "model_{}_baseline_{}_transition+speed_{}_seed_{}.txt".format(
            args.mode, baseline_int, speed, cur_seed
        ),
    )

    # Dataset names use integers, not floats (baseline_int already defined above)
    dset_name = "baseline_{}_transition+speed_{}_seed_{}".format(
        baseline_int, speed, cur_seed
    )
    
    #dset_name = os.path.join(args.dset, dset_name) Uncomment this to use a local copy of CCC

    model = models.resnet50(pretrained=True)
    model.to(device)
    # Only use DataParallel if CUDA is available and multiple GPUs exist
    if cuda_available and torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)

    assert args.mode in registery.get_options()
    if args.mode == "eta" or args.mode == "eata":
        loader = get_webds_loader(dset_name, args.dset if args.dset else None)
        model = registery.init(args.mode, model, loader, args.mode == "eta")
    else:
        model = registery.init(args.mode, model)

    test(model, dset_name, file_name=file_name, local_dset_path=args.dset if args.dset else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="tent", choices=registery.get_options()
    )
    parser.add_argument("--processind", type=int, default=0)
    parser.add_argument("--baseline", type=float, default=20)
    parser.add_argument("--logs", type=str)
    parser.add_argument("--dset", type=str)
    args = parser.parse_args()

    evaluate(args)

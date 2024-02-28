from collections import namedtuple
import time
import os
from dress_code_data import DressCodeDataLoader, DressCodeDataset

import torch
from tqdm import tqdm

# from utils import sem2onehot
import argparse


def get_conf(train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--category", default="all", type=str)
    parser.add_argument("--dataroot", type=str, default="/opt/dlami/nvme/DressCode")
    parser.add_argument("--data_pairs", default="{}_pairs")

    parser.add_argument(
        "--checkpoint_dir", type=str, default="", help="save checkpoint infos"
    )

    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-j", "--workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--display_count", type=int, default=1000)
    parser.add_argument(
        "--shuffle", default=True, action="store_true", help="shuffle input data"
    )

    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--radius", type=int, default=5)

    args = parser.parse_args()
    print(args)
    return args


def test_unpaired(dataloader, model, e, args):
    with tqdm(
        desc="Iteration %d - images extraction" % e,
        unit="it",
        total=len(dataloader.data_loader),
    ) as pbar:
        for step in range(0, len(dataloader.data_loader)):
            inputs = dataloader.next_batch()

            with torch.no_grad():
                image_name = inputs["im_name"]
                cloth_name = inputs["c_name"]
                image = inputs["image"].cuda()
                cloth = inputs["cloth"].cuda()
                cropped_cloth = inputs["im_cloth"].cuda()
                im_head = inputs["im_head"].cuda()
                pose_map = inputs["pose_map"].cuda()
                skeleton = inputs["skeleton"].cuda()
                im_pose = inputs["im_pose"].cuda()
                shape = inputs["shape"].cuda()
                parse_array = inputs["parse_array"].cuda()
                dense_labels = inputs["dense_labels"].cuda()
                dense_uv = inputs["dense_uv"].cuda()

                parse_array = sem2onehot(18, parse_array)

                # model here


def training_loop(dataloader, model, e, args):

    with tqdm(
        desc="Iteration %d - train" % e, unit="it", total=args.display_count
    ) as pbar:
        for step in range(0, args.display_count):
            inputs = dataloader.next_batch()

            image_name = inputs["im_name"]
            cloth_name = inputs["c_name"]
            image = inputs["image"].cuda()
            cloth = inputs["cloth"].cuda()
            cropped_cloth = inputs["im_cloth"].cuda()
            im_head = inputs["im_head"].cuda()
            pose_map = inputs["pose_map"].cuda()
            skeleton = inputs["skeleton"].cuda()
            im_pose = inputs["im_pose"].cuda()
            shape = inputs["shape"].cuda()
            parse_array = inputs["parse_array"].cuda()
            dense_labels = inputs["dense_labels"].cuda()
            dense_uv = inputs["dense_uv"].cuda()

            parse_array = sem2onehot(18, parse_array)

            # model here

            pbar.update()


def main_worker(args):

    # Dataset & Dataloader
    dataset_train = DressCodeDataset(
        args,
        dataroot_path=args.dataroot,
        phase="train",
        order="paired",
        size=(int(args.height), int(args.width)),
    )

    dataloader_train = DressCodeDataLoader(args, dataset_train, dist_sampler=False)

    dataset_test_unpaired = DressCodeDataset(
        args,
        dataroot_path=args.dataroot,
        phase="test",
        order="unpaired",
        size=(int(args.height), int(args.width)),
    )

    dataloader_test_unpaired = DressCodeDataLoader(
        args, dataset_test_unpaired, dist_sampler=False
    )

    # Instance here your model
    model = None
    with open("output.txt", "w") as file:
        print(dataloader_train.next_batch(), file=file)

    # Loop in epochs
    # for e in range(0, args.epochs):
    #     # Training loop
    #     training_loop(dataloader_train, model, e, args)

    #     # Test unpaired
    #     inputs = dataloader_test_unpaired.next_batch()
    #     test_unpaired(inputs, model, e, args)


if __name__ == "__main__":
    # Get argparser configuration
    args = get_conf()
    print(args.exp_name)

    # Call main worker
    main_worker(args)

import argparse
import random

import numpy as np
import torch
import yaml
from dataset import QMCompDataset
from evaluator import ThreeDEvaluator
from run import run

from model import SphereNet


def prepare_dataset(tr_filepath):
    # load data files
    tr_dataset = QMCompDataset(tr_filepath)
    print("train_dataset:", len(tr_dataset))

    target = "energy"
    tr_dataset._data["y"] = tr_dataset._data[target]
    tr_dataset.slices["y"] = tr_dataset.slices[target]

    # split validation set
    perm = torch.randperm(len(tr_dataset))
    train_num = int(0.8 * len(tr_dataset))
    train_idx = perm[:train_num]
    valid_idx = perm[train_num:]

    train_dataset = tr_dataset[train_idx]
    valid_dataset = tr_dataset[valid_idx]
    print("split dataset complete")
    print("tr:", len(train_dataset), "val:", len(valid_dataset))
    return train_dataset, valid_dataset


def prepare_model(args):
    model = SphereNet(
        energy_and_force=args.energy_and_force,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=128,
        out_channels=1,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_spherical=3,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    )

    return model


def run_qm(args, tr_filepath):
    # Dataset
    train_dataset, valid_dataset, test_dataset = prepare_dataset(tr_filepath)

    # Model, loss, and evaluation
    model = prepare_model(args)
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

    # Train and evaluate
    run3d = run()
    device = torch.device(args.device)

    log_dir = args.save_dir
    save_dir = args.save_dir
    restore_file = ""  # checkpoint file, use "" by default
    run3d.run(
        device,
        train_dataset,
        valid_dataset,
        # test_dataset,
        model,
        loss_func,
        evaluation,
        epochs=args.epochs,
        batch_size=args.batch_size,  # train batch size
        vt_batch_size=32,  # valid and test batch size
        lr=args.lr,
        lr_decay_factor=0.5,
        lr_decay_step_size=15,
        energy_and_force=args.energy_and_force,
        save_dir=save_dir,
        log_dir=log_dir,
        restore_file=restore_file,  # restore model from checkpoint file
    )


def freeze_rand(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run.\n")
    parser.add_argument("--config", type=str, default="config.yaml", help="config file to load")
    parser.add_argument("--trfile", type=str, help="train filepath")
    cmd_args = parser.parse_args()
    cfgs = yaml.load(open(cmd_args.config, "r"), Loader=yaml.FullLoader)
    args = argparse.Namespace(**cfgs)

    # or replace filepath here
    tr_filepath = cmd_args.trfile

    # run
    freeze_rand(args.seed)
    run_qm(args, tr_filepath)

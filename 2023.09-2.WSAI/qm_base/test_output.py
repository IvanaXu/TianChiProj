import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from dataset import QMCompDataset
from torch.autograd import grad
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import SphereNet


def output_stringfy_force(force):
    if isinstance(force, np.ndarray):
        f_list = force.tolist()
        f_list = [str(f) for f in f_list]
        force_str = ",".join(f_list)
    elif isinstance(force, list):
        f_list = [str(f) for f in force]
        force_str = ",".join(f_list)
    return force_str


def batch_force_reshape(batch_force, batch_data):
    batch = batch_data.batch  # index
    res = []
    idxes = list(set(batch_data["batch"].cpu().numpy()))
    for idx in idxes:
        mask = batch == idx
        res.append(batch_force[mask].detach().cpu().view(-1))
    return res


def test(model, data_loader, energy_and_force, device):
    model.eval()
    preds = []
    if energy_and_force:
        preds_force = []

    raw_forces = []
    for step, batch_data in enumerate(tqdm(data_loader)):
        batch_data = batch_data.to(device)
        out = model(batch_data)
        if energy_and_force:
            force = -grad(
                outputs=out,
                inputs=batch_data.pos,
                grad_outputs=torch.ones_like(out),
                create_graph=False,
                retain_graph=False,
            )[0]
            raw_forces.append(force.detach().cpu())
            pred_force = batch_force_reshape(force, batch_data)
            preds_force += pred_force
        preds.append(out.detach_().cpu())

    preds = torch.cat(preds, dim=0).squeeze()
    res = {"energy": preds, "force": preds_force}
    return res


def predict(args, ckp_file, te_filepath):
    device = torch.device(args.device)
    vt_batch_size = 32

    # data
    test_dataset = QMCompDataset(te_filepath)
    print("test_dataset:", len(test_dataset))

    target = "energy"
    test_dataset._data["y"] = test_dataset._data[target]
    test_dataset.slices["y"] = test_dataset.slices[target]

    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

    # model
    model = SphereNet(
        energy_and_force=True,
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
    model = model.to(device)

    # load checkpoint
    if ckp_file != "":
        checkpoint = torch.load(ckp_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # predict
    res = test(model, test_loader, energy_and_force=True, device=device)

    # output
    energy = [x.numpy() for x in res["energy"]]
    force = [output_stringfy_force(x.numpy()) for x in res["force"]]
    df = pd.DataFrame(data={"energy": energy, "force": force}, columns=["energy", "force"])
    df.to_csv("submission.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run.\n")
    parser.add_argument("--config", type=str, default="config.yaml", help="config file to load")
    parser.add_argument("--checkpoint", type=str, help="checkpoint filepath")
    parser.add_argument("--trfile", type=str, help="test filepath")
    cmd_args = parser.parse_args()
    cfgs = yaml.load(open(cmd_args.config, "r"), Loader=yaml.FullLoader)
    args = argparse.Namespace(**cfgs)

    # for example:
    # ckp_file = "./log/best_checkpoint.pt"
    # te_filepath = "./data/QMA_round1_test"
    # te_filepath = "./data/QMB_round1_test"
    ckp_file = cmd_args.checkpoint
    te_filepath = cmd_args.trfile
    predict(args, ckp_file, te_filepath)

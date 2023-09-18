import glob
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def read_moldata_file(filepath):
    filepath = Path(filepath)
    if filepath.suffix == ".pkl":
        mol_datas = pickle.load(open(filepath, "rb"))
    elif filepath.suffix == ".npy":
        mol_datas = np.load(filepath, allow_pickle=True).tolist()
    else:
        raise ValueError()
    return mol_datas


def collect_moldata(fpath_inpt):
    if isinstance(fpath_inpt, (tuple, list)):
        all_mol_datas = []
        for filepath in fpath_inpt:
            mol_datas = read_moldata_file(filepath)
            all_mol_datas += mol_datas
        return all_mol_datas
    elif isinstance(fpath_inpt, (str, Path)):
        mol_datas = read_moldata_file(fpath_inpt)
        return mol_datas
    else:
        raise TypeError(f"Unsupported filepath, got {type(fpath_inpt)}")


def read_data(filepath):
    assert isinstance(filepath, (str, Path))
    filepath = Path(filepath)
    filename, suffix = filepath.stem, filepath.suffix

    # check cache
    parent_dir = filepath.absolute().parent
    processed_fp = parent_dir / f"{filename}_processed.pt"
    if processed_fp.exists():
        data, slices = torch.load(processed_fp)
        return data, slices

    # no cached data
    print("processing data...")
    if suffix == "":
        # glob files by prefix name
        filepath_list = glob.glob(str(filepath) + "**.npy")
        mol_datas = collect_moldata(filepath_list)
    else:
        mol_datas = collect_moldata(filepath)

    # numpy to pyg Data
    all_datas = []
    for mol_data in tqdm(mol_datas):
        mol_name = mol_data["mol_name"]
        atom_count = torch.tensor(mol_data["atom_count"])
        bond_count = torch.tensor(mol_data["bond_count"])
        elements = torch.tensor(mol_data["elements"], dtype=torch.int64)  # atomic number
        coordinates = torch.tensor(mol_data["coordinates"], dtype=torch.float64)  # atomic coordinates
        # no label information in test dataset
        energy, force = torch.zeros(2)
        if "energy" in mol_data:
            energy = torch.tensor(mol_data["energy"], dtype=torch.float64)
        if "force" in mol_data:
            force = torch.tensor(mol_data["force"], dtype=torch.float64)
        m_data = Data(
            name=mol_name,
            atom_count=atom_count,
            bond_count=bond_count,
            z=elements,
            pos=coordinates,
            energy=energy,
            force=force,
        )
        all_datas.append(m_data)
    data, slices = InMemoryDataset.collate(all_datas)
    torch.save((data, slices), processed_fp)
    print(f"saved to {processed_fp.absolute()}")
    return data, slices


class QMCompDataset(InMemoryDataset):
    def __init__(self, filepath, double=False):
        super(QMCompDataset, self).__init__()
        self.double = double
        print("QMCompDataset: ===init===")
        self.data, self.slices = read_data(filepath)
        self.check_double(double)
        print(f"Dataset complete.")

    def check_double(self, double):
        if not double:
            self.data["pos"] = self.data["pos"].float()
            self.data["energy"] = self.data["energy"].float()
            self.data["force"] = self.data["force"].float()
        else:
            self.data["pos"] = self.data["pos"].double()
            self.data["energy"] = self.data["energy"].double()
            self.data["force"] = self.data["force"].double()

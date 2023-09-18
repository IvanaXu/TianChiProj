
# Prepare

1.Install basic environment
```bash
pip install -r requirements.txt
```

2.Follow `torch_geometric` installation instructions:
[https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

These packages are needed:
>torch-geometric  
>torch-cluster  
>torch-scatter  
>torch-sparse  
>torch-spline-conv

# Download Data
Then download and unzip data files to data directory. Data files should be like:
```bash
./data/QMB_round1_train_230725_0.npy
./data/QMB_round1_train_230725_1.npy
./data/QMB_round1_train_230725_2.npy
./data/QMB_round1_train_230725_3.npy
./data/QMB_round1_train_230725_4.npy

./data/QMB_round1_test_230725_0.npy
./data/QMB_round1_test_230725_1.npy
```

You can revise `prepare_dataset()` in `train.py` for custom training.

# Run and train

For channel A, use
```bash
python train.py --trfile ./data/QMA_round1_train_230725
```

For channel B, use
```bash
python train.py --trfile ./data/QMB_round1_train_230725
```

Change `trfile` in above command according to your data path.
If no file suffix is provided, `glob` will be used to gather all files with `.npy` extension based on `trfile`. See `read_data()` in `dataset.py` for more information.

By default only a few settings are available in `config.yaml`, feel free to expand it.

# Test and output

For channel A, use
```bash
python test_output.py --tefile ./data/QMA_round1_test_230725 --checkpoint ./log/best_checkpoint.pt
```

For channel B, use
```bash
python test_output.py --tefile ./data/QMB_round1_test_230725 --checkpoint ./log/best_checkpoint.pt
```

Replace with your own `tefile` and `checkpoint` path.
import math
import os.path as osp
from typing import Optional, Tuple

import numpy as np
import torch
from sacred import Ingredient

from rfsf.util import devices


dataset_ingredient = Ingredient("dataset")

_clustered_prefix = "clustered-"
_uci_prefix = "uci-"
_dataset_titles = {
    "sine": "Sine",
    "cosine": "Cosine",
    "heaviside": "Heaviside Step Function",
    "heavisine": "Combined Heaviside and Sine",
    "heavicosine": "Combined Heaviside and Cosine",
    "discontinuous_odd_cosine": "Discontinuous Cosine",
    "boston-housing": "Boston Housing",  # TODO: Find correct title and will dataset README.
    "concrete": "Concrete Compressive Strength",
    "energy": "Energy Efficiency",
    "kin8nm": "Kin8NM",  # TODO: Find correct title and will dataset README.
    "naval": "Condition Based Maintenance of Naval Propulsion Plants",
    "power-plant": "Combined Cycle Power Plant Data Set",
    "protein-tertiary-structure": "Physicochemical Properties of Protein Tertiary Structure",
    "wine-quality-red": "Wine Quality (Red)",
    "yacht": "Yacht Hydrodynamics",
}
_prefix_titles = {
    _clustered_prefix: "Clustered ",
    _uci_prefix: "UCI: ",
}


# noinspection PyUnusedLocal
@dataset_ingredient.config
def default_config():
    name = "sine"
    dataset_directory = "data/perm"
    uci_split_index = 0
    double_precision = False  # Can improve accuracy and can fix failing runs, must is very GPU-memory costly!


@dataset_ingredient.capture
def get_title(name: str) -> str:
    prefix = ""
    if name.startswith(_clustered_prefix):
        name = name[len(_clustered_prefix) :]
        prefix = _clustered_prefix
    elif name.startswith(_uci_prefix):
        name = name[len(_uci_prefix) :]
        prefix = _uci_prefix
    if prefix:
        prefix = _prefix_titles[prefix]
    if name in _dataset_titles:
        return prefix + _dataset_titles[name]
    assert False, f"unknown dataset '{name}'"


@dataset_ingredient.capture
def load_data(name: str, double_precision: bool, *, device: Optional[torch.device] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    if device is None:
        device = devices.cpu()

    print(f"Loading {name!r} dataset.")
    if name in (prefix + base_name for prefix in ("", _clustered_prefix) for base_name in ("sine", "cosine", "heaviside", "heavisine", "heavicosine", "discontinuous_odd_cosine")):
        func_name = name[len(_clustered_prefix) :] if name.startswith(_clustered_prefix) else name
        data = _load_dataset_similar_func(func_name, clustered=name.startswith(_clustered_prefix))
    elif name.startswith(_uci_prefix):
        uci_dataset_name = name[len(_uci_prefix) :]
        data = _load_uci_dataset(uci_dataset_name)
    else:
        assert False, f"unknown dataset {name!r}"
    # noinspection PyTypeChecker
    return tuple(tuple((da.double() if double_precision else da.float()).to(device) for da in dat) for dat in data)


def _load_dataset_similar_func(func_name: str, clustered: bool) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    if func_name == "sine":
        func = lambda x: torch.sin(2 * math.pi * x)
    elif func_name == "cosine":
        func = lambda x: torch.cos(2 * math.pi * x)
    elif func_name == "heaviside":
        func = lambda x: torch.maximum(torch.zeros_like(x), torch.sign(x))
    elif func_name == "heavisine":
        func = lambda x: torch.maximum(torch.zeros_like(x), torch.sign(x)) * torch.sin(2 * math.pi * x)
    elif func_name == "heavicosine":
        func = lambda x: torch.maximum(torch.zeros_like(x), torch.sign(x)) * torch.cos(2 * math.pi * x)
    elif func_name == "discontinuous_odd_cosine":
        func = lambda x: torch.sign(x) * torch.cos(2 * math.pi * x)
    else:
        assert False, f"unknown function name {func_name!r}"

    train_resolution = 0.05
    noise_var = 0.0001
    if clustered:
        interval_1_lo, interval_1_up = -1.0, -0.5
        interval_2_lo, interval_2_up = +0.5, +1.0
        interval_lo, interval_up = interval_1_lo, interval_2_up
        train_inputs = torch.cat([torch.arange(interval_1_lo, interval_1_up, train_resolution), torch.arange(interval_2_lo, interval_2_up, train_resolution)])
    else:
        interval_lo, interval_up = -0.5, 0.5
        train_inputs = torch.arange(interval_lo, interval_up, train_resolution)

    interval_width = interval_up - interval_lo
    test_resolution = 0.01

    test_inputs = torch.arange(interval_lo - interval_width / 2, interval_up + interval_width / 2, test_resolution)

    train_targets = func(train_inputs) + torch.randn(train_inputs.size()) * torch.tensor(math.sqrt(noise_var))
    test_targets = func(test_inputs)

    return (train_inputs.view((-1, 1)), train_targets), (test_inputs.view((-1, 1)), test_targets)


@dataset_ingredient.capture
def _load_uci_dataset(name: str, dataset_directory: str, uci_split_index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    uci_dir = f"{dataset_directory}/uci/{name}"
    assert osp.isdir(uci_dir), f"unknown UCI dataset {name}"

    n_splits = int(np.loadtxt(f"{uci_dir}/n_splits.txt"))
    assert 0 <= uci_split_index < n_splits, f"split index {uci_split_index} out of bounds; only {n_splits} available"

    data = np.loadtxt(f"{uci_dir}/data.txt", dtype=np.float32)
    features_dims = np.loadtxt(f"{uci_dir}/index_features.txt", dtype=int).reshape((-1,))
    target_dims = np.loadtxt(f"{uci_dir}/index_target.txt", dtype=int).reshape((-1,))
    train_indices = np.loadtxt(f"{uci_dir}/index_train_{uci_split_index}.txt", dtype=int)
    test_indices = np.loadtxt(f"{uci_dir}/index_test_{uci_split_index}.txt", dtype=int)

    train_data, test_data = data[train_indices], data[test_indices]
    train_inputs = train_data[:, features_dims]
    train_targets = train_data[:, target_dims][:, 0]  # TODO: Support multi-target regression!
    test_inputs = test_data[:, features_dims]
    test_targets = test_data[:, target_dims][:, 0]  # TODO: Support multi-target regression!

    return (torch.from_numpy(train_inputs), torch.from_numpy(train_targets)), (torch.from_numpy(test_inputs), torch.from_numpy(test_targets))

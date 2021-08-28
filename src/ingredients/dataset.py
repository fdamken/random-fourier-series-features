import math
from typing import Optional, Tuple

import numpy as np
import sklearn.model_selection
import torch
from sacred import Ingredient


dataset_ingredient = Ingredient("dataset")

_clustered_prefix = "clustered-"
_uci_prefix = "uci-"
_uci_number_of_outputs = {
    "boston-housing": 1,  # TODO: Find out if this is correct.
    "concrete": 1,
    "energy": 2,
    "kin8nm": 1,  # TODO: Find out if this is correct.
    "naval-propulsion-plant": 2,
    "power-plant": 1,
    "protein-tertiary-structure": 1,
    "wine-quality-red": 1,
    "wine-quality-white": 1,
    "yacht": 1,
}
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
    "wine-quality-white": "Wine Quality (White)",
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
    train_test_split_test_portion = 0.5
    train_test_split_seed = 42


@dataset_ingredient.capture
def get_title(name: str) -> str:
    prefix = ""
    if name.startswith(_clustered_prefix):
        name = name[len(_clustered_prefix):]
        prefix = _clustered_prefix
    elif name.startswith(_uci_prefix):
        name = name[len(_uci_prefix):]
        prefix = _uci_prefix
    if prefix:
        prefix = _prefix_titles[prefix]
    if name in _dataset_titles:
        return prefix + _dataset_titles[name]
    assert False, f"unknown dataset '{name}'"


@dataset_ingredient.capture
def load_data(name: str, *, device: Optional[torch.device] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    print(f"Loading {name!r} dataset.")
    if name in (prefix + base_name for prefix in ("", _clustered_prefix) for base_name in ("sine", "cosine", "heaviside", "heavisine", "heavicosine", "discontinuous_odd_cosine")):
        func_name = name[len(_clustered_prefix):] if name.startswith(_clustered_prefix) else name
        data = _load_dataset_similar_func(func_name, clustered=name.startswith(_clustered_prefix))
    elif name.startswith(_uci_prefix):
        uci_dataset_name = name[len(_uci_prefix):]
        data = _load_uci_dataset(uci_dataset_name)
    else:
        assert False, f"unknown dataset {name!r}"
    return data if device is None else tuple(tuple(da.to(device) for da in dat) for dat in data)


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

    return (train_inputs, train_targets), (test_inputs, test_targets)


@dataset_ingredient.capture
def _load_uci_dataset(name: str, dataset_directory: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    assert name in _uci_number_of_outputs.keys(), f"unknown UCI dataset {name}"
    number_of_outputs = _uci_number_of_outputs[name]
    file = f"{dataset_directory}/uci/{name}.csv"
    data = np.genfromtxt(file, dtype=np.float32, delimiter=",")
    train_data, test_data = _train_test_split(data)
    train_inputs = train_data[:, :-number_of_outputs]
    train_targets = train_data[:, :number_of_outputs][:, 0]  # TODO: Support multi-target regression!
    test_inputs = test_data[:, :-number_of_outputs]
    test_targets = test_data[:, :number_of_outputs][:, 0]  # TODO: Support multi-target regression!
    return (torch.from_numpy(train_inputs), torch.from_numpy(train_targets)), (torch.from_numpy(test_inputs), torch.from_numpy(test_targets))


@dataset_ingredient.capture
def _train_test_split(data: np.ndarray, train_test_split_test_portion: float, train_test_split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    return sklearn.model_selection.train_test_split(data, test_size=train_test_split_test_portion, random_state=train_test_split_seed)

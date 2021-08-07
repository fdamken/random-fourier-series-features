import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
from sacred import Ingredient


dataset_ingredient = Ingredient("dataset")

_clustered_prefix = "clustered-"
_clustered_title_prefix = "Clustered "
_dataset_titles = {
    "sine": "Sine",
    "cosine": "Cosine",
    "heaviside": "Heaviside Step Function",
    "heavisine": "Combined Heaviside and Sine",
    "heavicosine": "Combined Heaviside and Cosine",
    "discontinuous_odd_cosine": "Discontinuous Cosine",
}


# noinspection PyUnusedLocal
@dataset_ingredient.config
def default_config():
    name = "sine"


@dataset_ingredient.capture
def get_title(name: str) -> str:
    prefix = ""
    if name.startswith(_clustered_prefix):
        name = name[len(_clustered_prefix) :]
        prefix = _clustered_title_prefix
    if name in _dataset_titles:
        return prefix + _dataset_titles[name]
    assert False, f"unknown dataset '{name}'"


@lru_cache
@dataset_ingredient.capture
def load_data(name: str, *, device: Optional[torch.device] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    print(f"Loading {name!r} dataset.")
    if name in (prefix + base_name for prefix in ("", _clustered_prefix) for base_name in ("sine", "cosine", "heaviside", "heavisine", "heavicosine", "discontinuous_odd_cosine")):
        func_name = name[len(_clustered_prefix) :] if name.startswith(_clustered_prefix) else name
        data = _load_dataset_similar_func(func_name, clustered=name.startswith(_clustered_prefix))
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

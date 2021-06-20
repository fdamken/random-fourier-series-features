import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
from sacred import Ingredient


dataset_ingredient = Ingredient("dataset")

_dataset_titles = {
    "sine": "(Noisy) Sine",
    "heaviside": "Heaviside Step Function",
}


# noinspection PyUnusedLocal
@dataset_ingredient.config
def default_config():
    name = "sine"


@dataset_ingredient.capture
def get_title(name: str) -> str:
    if name in _dataset_titles:
        return _dataset_titles[name]
    assert False, f"unknown dataset '{name}'"


@lru_cache
@dataset_ingredient.capture
def load_data(name: str, *, device: Optional[torch.device] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    print(f"Loading '{name}' dataset.")
    if name == "sine":
        data = _load_dataset_sine()
    elif name == "heaviside":
        data = _load_dataset_heaviside()
    else:
        assert False, f"unknown dataset '{name}'"
    return data if device is None else tuple(tuple(da.to(device) for da in dat) for dat in data)


def _load_dataset_sine() -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    interval_lo, interval_up = -0.5, 0.5
    interval_width = interval_up - interval_lo
    train_resolution = 0.05
    test_resolution = 0.01
    noise_var = 0.0001
    func = lambda x: torch.sin(2 * math.pi * x)

    train_inputs = torch.arange(interval_lo, interval_up, train_resolution)
    train_targets = func(train_inputs) + torch.randn(train_inputs.size()) * torch.tensor(math.sqrt(noise_var))
    test_inputs = torch.arange(interval_lo - interval_width / 2, interval_up + interval_width / 2, test_resolution)
    test_targets = func(test_inputs)
    return (train_inputs, train_targets), (test_inputs, test_targets)


def _load_dataset_heaviside() -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    interval_lo, interval_up = -0.5, 0.5
    interval_width = interval_up - interval_lo
    train_resolution = 0.05
    test_resolution = 0.01
    noise_var = 0.0001
    func = lambda x: torch.maximum(torch.zeros_like(x), torch.sign(x))

    train_inputs = torch.arange(interval_lo, interval_up, train_resolution)
    train_targets = func(train_inputs)
    test_inputs = torch.arange(interval_lo - interval_width / 2, interval_up + interval_width / 2, test_resolution)
    test_targets = func(test_inputs)
    return (train_inputs, train_targets), (test_inputs, test_targets)

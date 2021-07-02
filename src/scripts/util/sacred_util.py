import os
import pickle
import re
from argparse import ArgumentParser
from functools import lru_cache
from typing import Any

import jsonpickle.ext.numpy
from sacred import Experiment, Ingredient
from torch import nn

from ingredients.dataset import dataset_ingredient
from rfsf.preprocessing.pre_processor import PreProcessor
from rfsf.util.tensor_util import unpickle_str


jsonpickle.ext.numpy.register_handlers()


def make_run_ingredient(base_dir: str):
    run_ingredient = Ingredient("run")

    @lru_cache
    def load_config() -> dict:
        return _load_jsonpickle(f"{base_dir}/config.json")

    @lru_cache
    def load_metrics() -> dict:
        return _load_jsonpickle(f"{base_dir}/metrics.json")

    @lru_cache
    def load_run() -> dict:
        return _load_jsonpickle(f"{base_dir}/run.json")

    @lru_cache
    def load_model() -> nn.Module:
        model = _load_pickle(f"{base_dir}/model.pkl")
        model.load_state_dict(unpickle_str(load_run()["result"]["model_state"]))
        return model

    @lru_cache
    def load_pre_processor() -> PreProcessor:
        return _load_pickle(f"{base_dir}/pre_processor.pkl")

    return run_ingredient, load_config, load_metrics, load_run, load_model, load_pre_processor


def load_experiment():
    parser = ArgumentParser()
    parser.add_argument("-b", "--base_dir", default="data/temp")
    parser.add_argument("-f", "--figures_dir", default="figures")
    parser.add_argument("-r", "--results_dir", default="results")
    parser.add_argument("-d", "--experiment_id", default="<latest>")
    args = parser.parse_args()
    figures_dir = f"{args.base_dir}/{args.figures_dir}"
    experiment_dir = f"{args.base_dir}/{args.results_dir}/{args.experiment_id}"
    match = re.match("^(.+)/<latest([+-][1-9][0-9]*)?>$", experiment_dir)
    if match:
        dirname = match.group(1)
        if dirname.strip() == "":
            raise Exception("Result container must not be root!")
        if match.group(2) is None:
            item = -1
        else:
            item = int(match.group(2)) - 1
        dirs = sorted([int(x) for x in os.listdir(dirname) if os.path.isdir(dirname + "/" + x) and x.isdigit()])
        experiment_dir = dirname + "/" + str(dirs[item])
    print(f"Reading results from {experiment_dir}.")

    if os.path.exists(figures_dir):
        assert os.path.isdir(figures_dir), f"{figures_dir=} must not exist or must be a directory"
    else:
        os.makedirs(figures_dir)

    run_ingredient, load_config, load_metrics, load_run, load_model, load_pre_processor = make_run_ingredient(experiment_dir)
    ex = Experiment(ingredients=[dataset_ingredient, run_ingredient])
    ex.add_config({"__experiment_dir": experiment_dir, "__figures_dir": figures_dir})
    config = load_config()
    ex.add_config(config)
    dataset_ingredient.add_config(config["dataset"])
    return ex, load_config, load_metrics, load_run, load_model, load_pre_processor


def _load_jsonpickle(path: str) -> dict:
    with open(path, "r") as f:
        return jsonpickle.loads(f.read())


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

import os
import os.path as osp
import pickle
import re
from argparse import ArgumentParser
from functools import lru_cache
from typing import Any, Iterator, Optional, Tuple

import jsonpickle.ext.numpy
from sacred import Experiment, Ingredient
from torch import nn

from ingredients.dataset import dataset_ingredient
from rfsf.pre_processing.pre_processor import PreProcessor
from rfsf.util.tensor_util import unpickle_str


jsonpickle.ext.numpy.register_handlers()


def make_run_ingredient(base_dir: str):
    run_ingredient = Ingredient("run")

    use_legacy_model_format = osp.isfile(f"{base_dir}/model.pkl")

    @lru_cache
    def load_config() -> dict:
        return _load_jsonpickle(f"{base_dir}/config.json")

    @lru_cache
    def load_metrics() -> dict:
        return _load_jsonpickle(f"{base_dir}/metrics.json")

    @lru_cache
    def load_run() -> dict:
        return _load_jsonpickle(f"{base_dir}/run.json")

    def load_model(step: Optional[int] = None) -> Optional[nn.Module]:
        if use_legacy_model_format:
            # LEGACY: Previously, the model state was saved in the result dict.
            model = _load_pickle(f"{base_dir}/model.pkl")
            if step is None:
                model_state_str = load_run()["result"]["model_state"]
            else:
                model_states = load_metrics()["model_state"]
                steps = model_states["steps"]
                state_dicts = model_states["values"]
                if step not in steps:
                    return None
                model_state_str = state_dicts[steps.index(step)]
            model.load_state_dict(unpickle_str(model_state_str))
        else:
            if step is None:
                model_path = f"{base_dir}/model-final.pkl"
            else:
                model_path = f"{base_dir}/model-{step}.pkl"
            if not osp.isfile(model_path):
                return None
            model = _load_pickle(model_path)
        return model

    def iterate_models() -> Iterator[Tuple[Optional[int], nn.Module]]:
        if use_legacy_model_format:
            steps = load_metrics()["model_state"]["steps"]
        else:
            steps = []
            for file_name in os.listdir(base_dir):
                match = re.match(r"^model-(\d+)\.pkl$", file_name)
                if match:
                    steps.append(int(match.group(1)))
            steps.sort()
        for step in steps:
            yield step, load_model(step)
        yield None, load_model(None)

    @lru_cache
    def load_pre_processor() -> PreProcessor:
        return _load_pickle(f"{base_dir}/pre_processor.pkl")

    return run_ingredient, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor


def load_experiment():
    parser = ArgumentParser()
    parser.add_argument("-b", "--base_dir", default="data/temp")
    parser.add_argument("-f", "--figures_dir", default="figures")
    parser.add_argument("-r", "--results_dir", default="results")
    parser.add_argument("-e", "--eval_dir", default="eval")
    parser.add_argument("-d", "--experiment_id", default="<latest>")
    parser.add_argument("-D", "--load_dumped_eval", action="store_true")
    args = parser.parse_args()
    figures_dir = f"{args.base_dir}/{args.figures_dir}"
    eval_dir = f"{args.base_dir}/{args.eval_dir}"
    experiment_dir = f"{args.base_dir}/{args.results_dir}/{args.experiment_id}"
    load_dumped_eval = args.load_dumped_eval
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

    if os.path.exists(eval_dir):
        assert os.path.isdir(eval_dir), f"{eval_dir=} must not exist or must be a directory"
    else:
        os.makedirs(eval_dir)

    run_ingredient, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor = make_run_ingredient(experiment_dir)
    ex = Experiment(ingredients=[dataset_ingredient, run_ingredient])
    ex.add_config({"__experiment_dir": experiment_dir, "__figures_dir": figures_dir, "__eval_dir": eval_dir, "__load_dumped_eval": load_dumped_eval})
    config = load_config()
    ex.add_config(config)
    dataset_ingredient.add_config(config["dataset"])
    return ex, load_config, load_metrics, load_run, load_model, iterate_models, load_pre_processor


def _load_jsonpickle(path: str) -> dict:
    with open(path, "r") as f:
        return jsonpickle.loads(f.read())


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

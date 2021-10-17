import warnings

import numpy
import wandb
from sacred.dependencies import get_digest
from sacred.observers import RunObserver


class WandbObserver(RunObserver):
    def __init__(self, **kwargs):
        self.run = wandb.init(**kwargs)
        self.resources = {}

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        self.__update_config(config)

    def completed_event(self, stop_time, result):
        if result:
            if not isinstance(result, tuple):
                result = (result,)
            for i, r in enumerate(result):
                if isinstance(r, (int, float)):
                    wandb.log({f"result_{i}": float(r)})
                elif isinstance(r, dict):
                    wandb.log(r)
                elif isinstance(r, object):
                    artifact = wandb.Artifact(f"result_{i}.pkl", type="result")
                    artifact.add_file(r)
                    self.run.log_artifact(artifact)
                elif isinstance(r, numpy.ndarray):
                    wandb.log({"result_{}".format(i): wandb.Image(r)})
                else:
                    warnings.warn(f"logging results does not support type '{type(r)}' results. Ignoring this result")

    def artifact_event(self, name, filename, metadata=None, content_type=None):
        if content_type is None:
            content_type = "file"
        artifact = wandb.Artifact(name, type=content_type)
        artifact.add_file(filename)
        self.run.log_artifact(artifact)

    def resource_event(self, filename):
        if filename not in self.resources:
            md5 = get_digest(filename)
            self.resources[filename] = md5

    def log_metrics(self, metrics_by_name, info):
        metrics_by_step = {}
        for name, metric_ptr in metrics_by_name.items():
            for step, value in sorted(zip(metric_ptr["steps"], metric_ptr["values"])):
                metrics = {}
                if step in metrics_by_step:
                    metrics = metrics_by_step[step]
                else:
                    metrics_by_step[step] = metrics
                if name in metrics:
                    warnings.warn(f"duplicated metric {name} for step {step}; overwriting with new metric")
                metrics[name] = value
        for step, metrics in sorted(metrics_by_step.items(), key=lambda v: v[0]):
            for name, value in metrics.items():
                if isinstance(value, numpy.ndarray):
                    wandb.log({name: wandb.Image(value)}, step=step)
                elif isinstance(value, (int, float)):
                    wandb.log({name: value}, step=step)
                else:
                    warnings.warn(f"metric {name} has unsupported type {type(value)} for logging to wandb")

    def __update_config(self, config):
        for k, v in config.items():
            self.run.config[k] = v
        self.run.config["resources"] = []

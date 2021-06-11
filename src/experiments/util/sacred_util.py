import os
import pickle
import tempfile
from typing import Any, NoReturn, Optional

import torch
from sacred.run import Run


def add_pickle_artifact(_run: Run, obj: Any, name: str, *, metadata=None, device: Optional[torch.device] = None) -> NoReturn:
    if metadata is None:
        metadata = {}
    file_ending = ".pkl"
    if device is not None:
        obj.cpu()
    _, f_path = tempfile.mkstemp(prefix=name, suffix=file_ending)
    try:
        with open(f_path, "wb") as f:
            pickle.dump(obj, f)
        _run.add_artifact(f_path, name + file_ending, metadata)
    finally:
        os.remove(f_path)
        if device is not None:
            obj.to(device)

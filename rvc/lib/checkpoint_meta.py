"""Read plain metadata from an RVC ``.pth`` checkpoint without torch.

A ``torch.save`` checkpoint is a zip holding ``<name>/data.pkl`` (the pickled
object graph) plus one file per tensor storage. The UI only needs scalar
fields such as ``speakers_id``, so the pickle is read with tensors replaced by
``None`` and their storages left unread: no torch import (the UI process
stays torch-free) and no weights in memory.

The unpickler resolves only containers and torch's tensor-rebuild helpers
(as stubs), so it is no less safe than ``torch.load(weights_only=True)``.
"""

import pickle
import zipfile

_ALLOWED = {("collections", "OrderedDict")}


def _stub(*_args, **_kwargs):
    return None


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        if module == "torch" or module.startswith("torch."):
            return _stub  # tensor rebuild helpers, storage and dtype classes
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        return None  # a tensor storage: never read


def read_checkpoint_metadata(path):
    """The checkpoint's top-level dict, with every tensor as ``None``.

    Raises ``ValueError`` for files that are not zip-format torch
    checkpoints (legacy pickles, ONNX, ...).
    """
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a zip-format torch checkpoint: {path}")
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
        if not pickles:
            raise ValueError(f"no data.pkl in {path}")
        with archive.open(pickles[0]) as f:
            data = _MetadataUnpickler(f).load()
    if not isinstance(data, dict):
        raise ValueError(f"unexpected checkpoint layout in {path}")
    return data

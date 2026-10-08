"""Load the exact author loss without importing the original training stack."""

import ast
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def load_author_loss(reference_dir):
    source = (Path(reference_dir) / "opsd_trainer.py").read_bytes()
    expected = "aa11fc2a3f3cd814da37db38c6fb2351cab7f98a5a572d808a9b75de16097c9d"
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("Reference source changed; expected OPSD revision ae7d2519")
    functions = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "generalized_jsd_loss"
    ]
    if len(functions) != 1:
        raise ValueError("Expected one author loss function")
    function = functions[0]
    function.decorator_list = []
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "original_opsd_loss", "exec"), namespace)
    return namespace[function.name]

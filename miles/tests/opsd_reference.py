"""Load the exact author loss without importing the original training stack."""

import ast
import hashlib
import importlib.util
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


def load_author_collator(reference_dir):
    source = Path(reference_dir) / "data_collator.py"
    expected = "5b96f3b3ae2f04e2b9dbf03509d3cd723c1b637d9362ab4697135d9f885b8381"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Reference collator changed; expected OPSD revision ae7d2519")
    spec = importlib.util.spec_from_file_location("original_opsd_collator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SelfDistillationDataCollator


def load_author_input_builder(reference_dir):
    """Execute the unchanged post-generation input/label block, without TRL.

    The checksum fixes its meaning. Only enclosing runtime setup, generation,
    logging and training are omitted; no statement inside this block is rewritten.
    """
    source = (Path(reference_dir) / "opsd_trainer.py").read_bytes()
    expected = "aa11fc2a3f3cd814da37db38c6fb2351cab7f98a5a572d808a9b75de16097c9d"
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("Reference trainer changed; expected OPSD revision ae7d2519")
    method = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "training_step"
    )
    start = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "student_prompt_len"
    )
    end = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "inputs['labels']"
    )
    function = ast.parse("def build(self, inputs, generated_ids, generated_attention_mask):\n    return inputs").body[0]
    function.body = method.body[start : end + 1] + function.body
    namespace = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "original_opsd_input_builder",
            "exec",
        ),
        namespace,
    )
    return namespace["build"]

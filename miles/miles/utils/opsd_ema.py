"""FP32 EMA state for a detached OPSD teacher and temporary matched forks."""

import math
from collections.abc import Mapping
from pathlib import Path

import torch


class OPSDEMA:
    def __init__(self, original: Mapping[str, torch.Tensor], *, decay: float):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.decay = decay
        self.updates = 0
        self.tensors = {name: tensor.detach().to(device="cpu", dtype=torch.float32).clone() for name, tensor in original.items() if tensor.is_floating_point()}
        if not self.tensors:
            raise ValueError("EMA requires floating model tensors")

    def _validate(self, weights: Mapping[str, torch.Tensor]) -> None:
        floating = {name for name, value in weights.items() if value.is_floating_point()}
        if floating != self.tensors.keys():
            raise ValueError("EMA parameter names changed")
        for name, average in self.tensors.items():
            value = weights[name]
            if value.shape != average.shape or not torch.isfinite(value).all():
                raise ValueError("EMA requires matching shapes and finite parameters")

    @torch.no_grad()
    def update(self, student: Mapping[str, torch.Tensor]) -> None:
        self._validate(student)
        for name, average in self.tensors.items():
            average.lerp_(student[name].detach().to(device="cpu", dtype=torch.float32), 1 - self.decay)
        self.updates += 1

    @torch.no_grad()
    def publish(self, scoring: Mapping[str, torch.Tensor]) -> None:
        self._validate(scoring)
        for name, average in self.tensors.items():
            scoring[name].copy_(average)

    @torch.no_grad()
    def relative_distance(self, weights: Mapping[str, torch.Tensor]) -> float:
        numerator = denominator = 0.0
        for name, average in self.tensors.items():
            value = weights[name].detach().to(device="cpu", dtype=torch.float32)
            numerator += float((average - value).square().sum(dtype=torch.float64))
            denominator += float(value.square().sum(dtype=torch.float64))
        return math.sqrt(numerator / max(denominator, 1e-30))

    def save(self, path: Path, *, student_sha256: str) -> None:
        """This is a paired teacher source, not an optimizer/recovery checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"decay": self.decay, "updates": self.updates, "tensors": self.tensors, "student_sha256": student_sha256}, temporary)
        temporary.replace(path)

    def load(self, path: Path, *, student_sha256: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["decay"] != self.decay or state["student_sha256"] != student_sha256:
            raise ValueError("EMA source belongs to a different decay or student fork")
        if not isinstance(state["updates"], int) or state["updates"] < 0:
            raise ValueError("Invalid EMA update count")
        self._validate(state["tensors"])
        if any(value.dtype != torch.float32 for value in state["tensors"].values()):
            raise ValueError("EMA source must preserve FP32 accumulation")
        self.tensors = state["tensors"]
        self.updates = state["updates"]

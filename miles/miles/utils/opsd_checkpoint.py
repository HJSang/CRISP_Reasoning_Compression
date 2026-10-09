"""Content identity for completed full-model OPSD evaluation checkpoints."""

import hashlib
import json
from pathlib import Path


def checkpoint_digest(directory: Path) -> str:
    """Hash weights and HF metadata without reading a model-sized buffer into RAM.

    The producer's completion marker must exist before enqueueing. Index entries
    are checked too, so a partial shard set cannot become a valid evaluation job.
    Tokenizer/config changes invalidate identity as well as weight changes.
    """
    if not (directory / ".complete").is_file():
        raise ValueError("Full OPSD evaluation checkpoint has no completion marker")
    if not (directory / "config.json").is_file():
        raise ValueError("Full OPSD evaluation checkpoint has no model config")
    weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
    if not weights or (directory / "adapter_config.json").exists():
        raise ValueError("Full OPSD evaluation requires complete model weights, not an adapter")
    for index in directory.glob("*.index.json"):
        for shard in json.loads(index.read_text())["weight_map"].values():
            if Path(shard).name != shard or not (directory / shard).is_file():
                raise ValueError("Full OPSD evaluation checkpoint is missing an indexed shard")
    hashes = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

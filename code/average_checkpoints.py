"""Average compatible research checkpoints without changing the evaluator format.

Example::

    python average_checkpoints.py --output runs/swa/checkpoint.pt \
        runs/p4/CAP-p02-s17/checkpoint-step-011500.pt \
        runs/p4/CAP-p02-s17/checkpoint-step-012000.pt

Only floating-point model tensors are accepted. Input checkpoints must share
protocol, implementation, configuration and seed. The output is an ordinary
``evaluate.py`` checkpoint with an additional provenance record.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from common import PROTOCOL


STEP_NAME = re.compile(r"checkpoint-step-(\d+)\.pt\Z")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_step(path, checkpoint, digest):
    """Recover the update number from a named step or verified run metadata."""
    match = STEP_NAME.fullmatch(path.name)
    if match:
        return int(match.group(1)), "filename"
    if isinstance(checkpoint.get("steps"), int) and not isinstance(checkpoint["steps"], bool):
        return checkpoint["steps"], "checkpoint"
    metrics_path = path.parent / "metrics.json"
    if path.name == "checkpoint.pt" and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("checkpoint_sha256") != digest:
            raise ValueError(f"Run metrics do not match checkpoint hash: {path}")
        for field in ("protocol", "implementation", "config", "seed", "train_tokens"):
            if metrics.get(field) != checkpoint.get(field):
                raise ValueError(f"Run metrics disagree on {field}: {path}")
        step = metrics.get("steps")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"Run metrics have no valid step: {metrics_path}")
        return step, "verified_metrics"
    raise ValueError(f"Cannot establish training step for {path}; use original run files")


def load_source(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    for field in ("protocol", "implementation", "config", "seed", "train_tokens", "model"):
        if field not in checkpoint:
            raise ValueError(f"Checkpoint lacks {field}: {path}")
    if checkpoint["protocol"] != PROTOCOL:
        raise ValueError(f"Wrong protocol: {path}")
    if not isinstance(checkpoint["implementation"], str) or not checkpoint["implementation"]:
        raise ValueError(f"Invalid implementation: {path}")
    if not isinstance(checkpoint["config"], dict):
        raise ValueError(f"Invalid config: {path}")
    try:
        config_key = json.dumps(checkpoint["config"], sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config is not plain finite JSON: {path}") from exc
    for field in ("seed", "train_tokens"):
        value = checkpoint[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Invalid {field}: {path}")
    state = checkpoint["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Model state is empty or invalid: {path}")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid state entry {name!r}: {path}")
        if tensor.layout != torch.strided or not torch.is_floating_point(tensor):
            raise ValueError(f"Non-floating or non-dense state entry {name}: {path}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite state entry {name}: {path}")
    if "token.weight" in state and "head.weight" in state:
        if not torch.equal(state["token.weight"], state["head.weight"]):
            raise ValueError(f"Tied token/head embeddings disagree: {path}")
    step, step_source = source_step(path, checkpoint, digest)
    return checkpoint, config_key, {
        "name": f"{path.parent.name}/{path.name}",
        "sha256": digest,
        "step": step,
        "step_source": step_source,
        "train_tokens": checkpoint["train_tokens"],
    }


def average_checkpoints(inputs, output):
    """Write a direct-evaluation checkpoint and return its provenance summary."""
    paths = [Path(path).resolve() for path in inputs]
    output = Path(output)
    output = output.parent.resolve() / output.name
    if len(paths) < 2:
        raise ValueError("At least two checkpoints are required")
    if len(set(paths)) != len(paths):
        raise ValueError("A checkpoint path appears more than once")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    sources = [load_source(path) for path in paths]
    first, config_key, _ = sources[0]
    reference_state = first["model"]
    reference_keys = set(reference_state)
    for path, (checkpoint, key, _) in zip(paths[1:], sources[1:]):
        for field in ("protocol", "implementation", "seed"):
            if checkpoint[field] != first[field]:
                raise ValueError(f"Mismatched {field}: {path}")
        if key != config_key:
            raise ValueError(f"Mismatched config: {path}")
        state = checkpoint["model"]
        if set(state) != reference_keys:
            raise ValueError(f"Mismatched state keys: {path}")
        for name, reference in reference_state.items():
            tensor = state[name]
            if tensor.shape != reference.shape:
                raise ValueError(f"Mismatched shape for {name}: {path}")
            if tensor.dtype != reference.dtype:
                raise ValueError(f"Mismatched dtype for {name}: {path}")

    count = len(sources)
    averaged = {}
    for name, reference in reference_state.items():
        accumulator = torch.zeros(reference.shape, dtype=torch.float64)
        for checkpoint, _, _ in sources:
            accumulator.add_(checkpoint["model"][name].to(torch.float64))
        averaged[name] = (accumulator / count).to(reference.dtype)
    source_records = [record for _, _, record in sources]
    payload = {
        "protocol": first["protocol"],
        "implementation": first["implementation"],
        "config": copy.deepcopy(first["config"]),
        "model": averaged,
        "seed": first["seed"],
        "train_tokens": max(record["train_tokens"] for record in source_records),
        "steps": max(record["step"] for record in source_records),
        "averaging": {"method": "equal_weight_state_dict", "sources": source_records},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{output.name}.",
                                         suffix=".tmp", dir=output.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Link a complete file atomically. A pre-existing destination makes
        # os.link fail rather than replacing the checkpoint.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "sha256": file_sha256(output),
            "seed": payload["seed"], "steps": payload["steps"],
            "train_tokens": payload["train_tokens"], "sources": source_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        summary = average_checkpoints(args.inputs, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

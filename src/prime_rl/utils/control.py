"""Commit records the trainer writes under ``<output_dir>/control`` once it has saved the step a
control request applied to. The orchestrator's control API and the launcher read them."""

import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ControlAction = Literal["checkpoint", "pause"]


class ControlCommit(BaseModel):
    id: str
    action: ControlAction
    step: int


def get_control_dir(output_dir: Path) -> Path:
    return output_dir / "control"


def write_commit(control_dir: Path, commit: ControlCommit) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=control_dir, prefix=f"{commit.id}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(commit.model_dump_json())
    os.replace(tmp_name, control_dir / f"{commit.id}.json")


def read_commit(control_dir: Path, request_id: str) -> ControlCommit | None:
    path = control_dir / f"{request_id}.json"
    if not path.exists():
        return None
    return ControlCommit.model_validate_json(path.read_text())


def paused_since(control_dir: Path, since: float) -> bool:
    """Whether a pause was committed at or after ``since`` (a ``time.time()`` timestamp)."""
    return any(
        path.stat().st_mtime >= since and ControlCommit.model_validate_json(path.read_text()).action == "pause"
        for path in control_dir.glob("*.json")
    )

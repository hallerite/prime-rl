"""Run control API: ask the run to checkpoint or pause at the next step it ships, then follow the
request until the trainer commits it."""

import asyncio
import contextlib
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel

from prime_rl.configs.orchestrator import ControlConfig
from prime_rl.utils.control import ControlAction, read_commit

ControlState = Literal["pending", "accepted", "committed"]

MAX_WAIT_SECONDS = 60


class ControlRequest(BaseModel):
    action: ControlAction


class ControlRecord(BaseModel):
    id: str
    action: ControlAction
    state: ControlState
    step: int | None = None
    """The step the request applies to, set once the orchestrator ships it."""


class ControlConflict(Exception):
    pass


class ControlPlane:
    """Requests submitted through the API. The orchestrator applies a pending request to the next
    batch it ships."""

    def __init__(self, control_dir: Path) -> None:
        self.control_dir = control_dir
        self.records: dict[str, ControlRecord] = {}
        self.pending: ControlRecord | None = None
        self.pausing = False
        self.reported: set[str] = set()

    def submit(self, action: ControlAction) -> ControlRecord:
        if self.pausing:
            raise ControlConflict("The run is pausing")
        if self.pending is not None:
            raise ControlConflict(f"Control request {self.pending.id} is still pending")
        record = ControlRecord(id=uuid.uuid4().hex, action=action, state="pending")
        self.records[record.id] = self.pending = record
        return record

    def take(self) -> ControlRecord | None:
        record, self.pending = self.pending, None
        return record

    def accept(self, record: ControlRecord, step: int) -> None:
        record.state, record.step = "accepted", step
        self.pausing = self.pausing or record.action == "pause"

    def get(self, request_id: str) -> ControlRecord | None:
        record = self.records.get(request_id)
        if record is not None and record.state == "accepted" and read_commit(self.control_dir, request_id):
            record.state = "committed"
        return record

    def report(self, request_id: str) -> ControlRecord | None:
        """``get`` for an API client, remembering which commits a client has seen."""
        record = self.get(request_id)
        if record is not None and record.state == "committed":
            self.reported.add(request_id)
        return record


def create_app(plane: ControlPlane) -> FastAPI:
    app = FastAPI(title="prime-rl control")

    @app.post("/v1/control", status_code=status.HTTP_202_ACCEPTED)
    async def submit(request: ControlRequest) -> ControlRecord:
        try:
            return plane.submit(request.action)
        except ControlConflict as e:
            raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e

    @app.get("/v1/control/{request_id}")
    async def get(request_id: str, wait: float = Query(0, ge=0, le=MAX_WAIT_SECONDS)) -> ControlRecord:
        """``wait``: seconds to hold the request open until the trainer commits it."""
        record = plane.report(request_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown control request {request_id}")
        deadline = time.monotonic() + wait
        while record.state != "committed" and time.monotonic() < deadline:
            await asyncio.sleep(min(1.0, deadline - time.monotonic()))
            record = plane.report(request_id)
        return record

    return app


class ControlServer(uvicorn.Server):
    """Serves on the orchestrator's event loop; the orchestrator keeps its own signal handling."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def setup_control_server(plane: ControlPlane, config: ControlConfig) -> ControlServer:
    return ControlServer(uvicorn.Config(create_app(plane), host=config.host, port=config.port, log_level="warning"))

"""
Sequential Model Scheduler & Request Queue.
Coordinates single-model GPU execution, concurrency for active models,
grouped queue scheduling, and idle auto-unloading.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import settings
from models_catalog import get_model
from vllm_controller import vllm_controller

logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    model_id: str
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)


class SequentialScheduler:
    def __init__(self):
        self._active_requests: int = 0
        self._queue: List[QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._idle_task: Optional[asyncio.Task] = None
        self._idle_started_at: Optional[float] = None
        self._is_switching: bool = False

    @property
    def active_model(self) -> Optional[str]:
        return vllm_controller.current_model_id

    @property
    def active_request_count(self) -> int:
        return self._active_requests

    @property
    def queued_request_count(self) -> int:
        return len(self._queue)

    def get_status(self) -> dict:
        timer_active = bool(self._idle_task and not self._idle_task.done() and vllm_controller.is_running)
        idle_remaining = None
        if timer_active and self._idle_started_at is not None and settings.IDLE_UNLOAD_TIMEOUT > 0:
            elapsed = time.time() - self._idle_started_at
            idle_remaining = max(0, int(settings.IDLE_UNLOAD_TIMEOUT - elapsed))

        return {
            "active_model": self.active_model,
            "is_vllm_running": vllm_controller.is_running,
            "active_requests": self._active_requests,
            "queued_requests": len(self._queue),
            "is_switching": self._is_switching,
            "idle_timeout_seconds": settings.IDLE_UNLOAD_TIMEOUT,
            "idle_timer_active": timer_active,
            "idle_seconds_remaining": idle_remaining,
        }

    def _cancel_idle_timer(self):
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None
        self._idle_started_at = None

    async def unload(self) -> None:
        """Explicitly cancel idle countdown and unload active model from GPU."""
        async with self._lock:
            self._cancel_idle_timer()
            await vllm_controller.stop()

    def _start_idle_timer(self):
        self._cancel_idle_timer()
        if settings.IDLE_UNLOAD_TIMEOUT > 0:
            self._idle_started_at = time.time()
            self._idle_task = asyncio.create_task(self._idle_worker())

    async def _idle_worker(self):
        try:
            logger.info("Starting idle timer (%ds) for model '%s'", settings.IDLE_UNLOAD_TIMEOUT, self.active_model)
            await asyncio.sleep(settings.IDLE_UNLOAD_TIMEOUT)
            async with self._lock:
                if self._active_requests == 0 and len(self._queue) == 0:
                    logger.info("Idle timeout reached (%ds). Unloading model '%s' from GPU.", settings.IDLE_UNLOAD_TIMEOUT, self.active_model)
                    await vllm_controller.stop()
        except asyncio.CancelledError:
            pass
        finally:
            self._idle_started_at = None

    async def acquire_slot(self, requested_model_id: str) -> None:
        """
        Request inference execution slot.
        If the model is already running, grants immediate concurrency.
        Otherwise, queues request and initiates sequential model switch.
        """
        model_entry = get_model(requested_model_id)
        if not model_entry:
            raise ValueError(f"Model '{requested_model_id}' is not registered in models catalog.")

        async with self._lock:
            self._cancel_idle_timer()

            # Fast path: requested model is already running and ready, and no switch is happening
            if (
                vllm_controller.is_running
                and self.active_model == requested_model_id
                and not self._is_switching
                and len(self._queue) == 0
            ):
                self._active_requests += 1
                logger.info(
                    "Granted concurrent slot for active model '%s' (active: %d)",
                    requested_model_id,
                    self._active_requests,
                )
                return

            # Otherwise, must queue
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            queued = QueuedRequest(model_id=requested_model_id, future=future)
            self._queue.append(queued)

            logger.info(
                "Enqueued request for model '%s' (queue depth: %d, active: %d on '%s')",
                requested_model_id,
                len(self._queue),
                self._active_requests,
                self.active_model,
            )

            # Trigger scheduler pump
            asyncio.create_task(self._pump_queue())

        # Wait for the future to be resolved when slot is granted
        await future

    async def release_slot(self, model_id: str) -> None:
        """Called when an inference request completes."""
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            logger.info(
                "Released slot for model '%s' (active remaining: %d, queued: %d)",
                model_id,
                self._active_requests,
                len(self._queue),
            )

            if self._active_requests == 0:
                if len(self._queue) > 0:
                    # Pump queue to switch or dispatch
                    asyncio.create_task(self._pump_queue())
                else:
                    # No active requests and empty queue -> start idle countdown
                    self._start_idle_timer()

    async def _pump_queue(self) -> None:
        """Processes queued requests, performing model switching when active requests drain."""
        async with self._lock:
            if self._is_switching or len(self._queue) == 0:
                return

            # Check if there are queued requests that match the currently loaded model
            if vllm_controller.is_running and self.active_model is not None:
                same_model_requests = [q for q in self._queue if q.model_id == self.active_model]
                if same_model_requests:
                    # Dispatch same-model requests immediately
                    for q in same_model_requests:
                        self._queue.remove(q)
                        self._active_requests += 1
                        if not q.future.done():
                            q.future.set_result(True)
                    logger.info("Dispatched %d queued requests to active model '%s'", len(same_model_requests), self.active_model)
                    return

            # If requests are still active on current model, wait for them to finish
            if self._active_requests > 0:
                logger.info(
                    "Waiting for %d active requests on '%s' to complete before switching...",
                    self._active_requests,
                    self.active_model,
                )
                return

            # Active requests are 0. Select next model using FIFO + same-model grouping
            self._is_switching = True

        # Perform switch outside main lock to allow status queries, but guarded by _is_switching
        target_model_id = None
        target_requests: List[QueuedRequest] = []
        try:
            async with self._lock:
                if len(self._queue) == 0:
                    self._is_switching = False
                    return

                # Head of FIFO queue
                head = self._queue[0]
                target_model_id = head.model_id

                # Group all queued requests for the same target model
                target_requests = [q for q in self._queue if q.model_id == target_model_id]
                for q in target_requests:
                    self._queue.remove(q)

            logger.info(
                "Initiating model transition -> '%s' (batch size: %d, remaining queue: %d)",
                target_model_id,
                len(target_requests),
                len(self._queue),
            )

            model_entry = get_model(target_model_id)
            if not model_entry:
                raise ValueError(f"Model entry for '{target_model_id}' not found")

            # Start vLLM with the new model
            success = await vllm_controller.start(model_entry)
            if not success:
                err_detail = vllm_controller.last_error or "Check server console logs for details."
                raise RuntimeError(f"Failed to initialize vLLM engine for model '{target_model_id}': {err_detail}")

            # Dispatch all requests for this target model
            async with self._lock:
                self._active_requests += len(target_requests)
                for q in target_requests:
                    if not q.future.done():
                        q.future.set_result(True)

            logger.info("Model '%s' successfully loaded and %d requests dispatched.", target_model_id, len(target_requests))

        except Exception as exc:
            logger.error("Error switching to model '%s': %s", target_model_id, exc)
            for q in target_requests:
                if not q.future.done():
                    q.future.set_exception(exc)
        finally:
            async with self._lock:
                self._is_switching = False
                if self._active_requests == 0 and len(self._queue) > 0:
                    asyncio.create_task(self._pump_queue())


# Global singleton instance
scheduler = SequentialScheduler()

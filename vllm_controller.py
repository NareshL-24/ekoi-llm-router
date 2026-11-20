"""
vLLM Subprocess Controller.
Manages the lifecycle of the single vLLM inference process on the GPU server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from config import settings
from models_catalog import ModelEntry, resolve_vllm_command

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> dict:
    """Read GPU memory stats using NVML if available."""
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "available": True,
                "total_mb": round(mem_info.total / (1024**2), 1),
                "used_mb": round(mem_info.used / (1024**2), 1),
                "free_mb": round(mem_info.free / (1024**2), 1),
                "utilization_pct": round((mem_info.used / mem_info.total) * 100, 1),
            }
    except Exception:
        pass
    return {"available": False}


class VLLMController:
    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.current_model_id: Optional[str] = None
        self.last_error: Optional[str] = None
        self._recent_logs: list[str] = []
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self, model: ModelEntry) -> bool:
        """Launch vLLM subprocess for the specified model and wait until healthy."""
        async with self._lock:
            self.last_error = None
            self._recent_logs = []
            if self.is_running:
                if self.current_model_id == model.id:
                    logger.info("Model '%s' is already active and healthy.", model.id)
                    return True
                logger.info("Shutting down current model '%s' before starting '%s'", self.current_model_id, model.id)
                await self._stop_locked()

            if settings.MOCK_MODE:
                logger.info("[MOCK_MODE] Simulating vLLM startup for model '%s'", model.id)
                await asyncio.sleep(0.5)
                self.current_model_id = model.id
                return True

            cmd = resolve_vllm_command(model)
            logger.info("Launching vLLM subprocess for model '%s' on port %d...", model.id, settings.VLLM_INTERNAL_PORT)
            logger.info("Command: %s", " ".join(cmd))

            gpu_before = get_gpu_memory_info()
            if gpu_before.get("available"):
                logger.info("GPU VRAM before load: %s MB used / %s MB free", gpu_before["used_mb"], gpu_before["free_mb"])

            env = os.environ.copy()
            venv_bin = str(Path(sys.executable).parent)
            env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
            env["PYTHONUNBUFFERED"] = "1"

            try:
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
                self.current_model_id = model.id

                # Start background log reader
                asyncio.create_task(self._stream_vllm_logs(self.process))

                # Wait for health endpoint
                healthy = await self._wait_for_health(settings.VLLM_STARTUP_TIMEOUT)
                if not healthy:
                    logger.error("vLLM failed to become healthy within %ds. Killing process.", settings.VLLM_STARTUP_TIMEOUT)
                    await self._stop_locked()
                    return False

                gpu_after = get_gpu_memory_info()
                if gpu_after.get("available"):
                    logger.info("vLLM is ready. GPU VRAM: %s MB used / %s MB free", gpu_after["used_mb"], gpu_after["free_mb"])

                return True

            except Exception as exc:
                self.last_error = f"Failed to launch vLLM process: {exc}"
                logger.error("Failed to start vLLM for model '%s': %s", model.id, exc)
                await self._stop_locked()
                return False

    async def _stream_vllm_logs(self, proc: asyncio.subprocess.Process):
        """Stream stdout/stderr from vLLM subprocess to python logger."""
        try:
            if proc.stdout is not None:
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded:
                        self._recent_logs.append(decoded)
                        if len(self._recent_logs) > 50:
                            self._recent_logs.pop(0)
                        logger.info("[vLLM] %s", decoded)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("vLLM log stream ended: %s", exc)

    async def _wait_for_health(self, timeout_seconds: int) -> bool:
        """Poll vLLM health endpoint until 200 OK."""
        health_url = f"{settings.vllm_base_url}/health"
        start_time = time.time()
        logger.info("Waiting for vLLM health check at %s (timeout: %ds)...", health_url, timeout_seconds)

        async with httpx.AsyncClient(timeout=3.0) as client:
            while time.time() - start_time < timeout_seconds:
                if not self.is_running:
                    exit_code = self.process.returncode if self.process else "None"
                    recent = "\n".join(self._recent_logs[-15:])
                    self.last_error = f"vLLM subprocess exited prematurely with code {exit_code}. Logs:\n{recent}"
                    logger.error("%s", self.last_error)
                    return False
                try:
                    res = await client.get(health_url)
                    if res.status_code == 200:
                        logger.info("vLLM health check passed in %.1fs!", time.time() - start_time)
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                await asyncio.sleep(1.0)

        recent = "\n".join(self._recent_logs[-15:])
        self.last_error = f"vLLM failed to become healthy within {timeout_seconds}s. Logs:\n{recent}"
        logger.error("%s", self.last_error)
        return False

    async def stop(self) -> None:
        """Public stop method with lock."""
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Terminate the vLLM subprocess and release GPU VRAM."""
        if settings.MOCK_MODE:
            logger.info("[MOCK_MODE] Simulated vLLM shutdown for '%s'", self.current_model_id)
            self.current_model_id = None
            self.process = None
            return

        if not self.is_running or self.process is None:
            self.current_model_id = None
            self.process = None
            return

        logger.info("Stopping vLLM subprocess (PID %s) for model '%s'...", self.process.pid, self.current_model_id)
        pid = self.process.pid

        try:
            if sys.platform != "win32":
                self.process.send_signal(signal.SIGTERM)
            else:
                self.process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            for _ in range(20):
                if self.process.returncode is not None:
                    break
                await asyncio.sleep(0.5)

            if self.process.returncode is None:
                logger.warning("vLLM process did not exit gracefully; forcing SIGKILL...")
                if sys.platform != "win32":
                    self.process.send_signal(signal.SIGKILL)
                else:
                    self.process.kill()
                await self.process.wait()

            logger.info("vLLM subprocess (PID %s) terminated successfully.", pid)

        except Exception as exc:
            logger.error("Error stopping vLLM subprocess: %s", exc)
        finally:
            self.process = None
            self.current_model_id = None

            # Check released VRAM
            gpu_after = get_gpu_memory_info()
            if gpu_after.get("available"):
                logger.info("GPU VRAM after shutdown: %s MB used / %s MB free", gpu_after["used_mb"], gpu_after["free_mb"])


# Singleton instance
vllm_controller = VLLMController()

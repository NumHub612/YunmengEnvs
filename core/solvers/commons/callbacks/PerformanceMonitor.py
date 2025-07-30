# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Performance Monitor Callback.
"""
from core.solvers.interfaces import ISolverCallback, ISolver
from core.numerics.mesh import Mesh
from configs.settings import logger

import subprocess
import os
import json
import platform
import psutil
import GPUtil


class PerformanceMonitor(ISolverCallback):
    """
    Performance Monitor Callback.

    This callback monitors the performance of the solver, including:
    - Time usage
    - CPU usage
    - GPU usage
    - Memory usage
    - etc.
    """

    @classmethod
    def get_name(self) -> str:
        return "PerformanceMonitor"

    @property
    def id(self) -> str:
        return self._id

    def __init__(self, id: str, output_dir: str, interval: int = 1):
        """
        Constructor.
        """
        self._id = id
        self._process = os.getpid()
        self._perf_process = None
        self._interval = interval
        self._step = 0

        # start the performance monitoring process.
        profile = f"{id}_profile.svg"
        profile = os.path.join(output_dir, profile)
        self._perf_process = subprocess.Popen(
            ["py-spy", "record", "-o", profile, "--pid", str(self._process)]
        )
        logger.info(f"Start performance monitoring {id} process {self._process}")

    def _get_cpu_usage(self) -> dict:
        """
        Get the CPU usage of the current process.
        """
        return {
            "cpu_usage": psutil.Process(self._process).cpu_percent(),
        }

    def _get_memory_usage(self) -> dict:
        """
        Get the memory usage of the current process.
        """
        return {
            "memory_usage": psutil.Process(self._process).memory_percent(),
        }

    def _get_gpu_usage(self) -> dict:
        """
        Get the GPU usage.
        """
        total_memories, used_memories = [], []
        for gpu in GPUtil.getGPUs():
            total_memories.append(gpu.memoryTotal)
            used_memories.append(gpu.memoryUsed)

        if len(GPUtil.getGPUs()) > 0:
            memory_usages = [
                used / total for used, total in zip(used_memories, total_memories)
            ]
        else:
            memory_usages = []

        return {
            "memory_usage": memory_usages,
            "used_memory": used_memories,
        }

    def _get_platform_info(self) -> dict:
        """
        Get the platform information.
        """
        return {
            "version": platform.version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
        }

    def _collect_running_info(self) -> dict:
        """
        Collect the running information.
        """
        status = {}
        status.update(self._get_cpu_usage())
        status.update(self._get_memory_usage())
        status.update(self._get_gpu_usage())
        status.update({"step": self._step})
        return status

    def setup(self, solver: ISolver, mesh: Mesh, **kwargs):
        # Log the platform information.
        platform_info = self._get_platform_info()
        logger.info(json.dumps(platform_info))

    def cleanup(self, *args, **kwargs):
        self._step = 0
        if self._perf_process is not None:
            self._perf_process.send_signal(subprocess.signal.SIGINT)
            self._perf_process.wait()
            logger.info("Stop performance monitoring process")
        self._perf_process = None
        self._process = None

    def on_task_begin(self, *args, **kwargs):
        pass

    def on_task_end(self, *args, **kwargs):
        if self._perf_process is not None:
            self._perf_process.send_signal(subprocess.signal.SIGINT)
            self._perf_process.wait()
            logger.info("Stop performance monitoring process")
        pass

    def on_step_begin(self, *args, **kwargs):
        pass

    def on_step(self, *args, **kwargs):
        self._step += 1
        if self._step % self._interval != 0:
            return

        status = self._collect_running_info()
        logger.info(json.dumps(status))

    def on_step_end(self, *args, **kwargs):
        pass

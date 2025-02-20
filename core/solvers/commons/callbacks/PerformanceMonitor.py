# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Join us, share your ideas!  

Performance Monitor Callback.
"""
from core.solvers.interfaces import ISolverCallback
from core.numerics.mesh import Mesh
from configs.settings import logger

import subprocess
import os
import json
import platform
import logging
import uuid
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

    def __init__(self, log_id: str, log_file: str, interval: int = 1):
        """
        Constructor.
        """
        self._log_id = log_id
        self._log_file = log_file
        self._log = self._init_logging(log_id, log_file)

        self._process = os.getpid()
        self._perf_process = None
        self._interval = interval
        self._step = 0

        # start the performance monitoring process.
        log_dir = os.path.dirname(self._log_file)
        profile = f"{self._log_id}_profile.svg"
        profile = os.path.join(log_dir, profile)
        self._perf_process = subprocess.Popen(
            ["py-spy", "record", "-o", profile, "--pid", str(self._process)]
        )
        self._log.info(f"Start performance monitoring process")

    def _init_logging(self, log_id: str, log_file: str):
        """
        Initialize the unic logging.
        """
        if logging.getLogger(log_id).hasHandlers():
            self._log_id = log_id + "_" + str(uuid.uuid4())
            logger.warning(
                f"Log id {log_id} existed, use new log id {self._log_id} instead."
            )

        log = logging.getLogger(self._log_id)
        log.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(process)d][%(thread)d]: %(message)s"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        return log

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
            "used_memory": used_memories,
            "memory_usage": memory_usages,
        }

    def _get_platform_info(self) -> dict:
        """
        Get the platform information.
        """
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        }

    def _collect_running_info(self, solver_status: dict) -> dict:
        """
        Collect the running information.
        """
        status = solver_status
        status.update(self._get_cpu_usage())
        status.update(self._get_memory_usage())
        status.update(self._get_gpu_usage())
        status.update({"step": self._step})
        return status

    def setup(self, solver_meta: dict, mesh: Mesh):
        # Log the platform information.
        platform_info = self._get_platform_info()
        self._log.info(json.dumps(platform_info))

        # Log the solver information.
        self._log.info(json.dumps(solver_meta))

        # Log the mesh information.
        geom = mesh.get_geom_assistant()
        mesh_info = {
            "dimension": mesh.dimension,
            "is_orthogonal": mesh.is_orthogonal,
            "node_count": mesh.node_count,
            "face_count": mesh.face_count,
            "cell_count": mesh.cell_count,
            "face_area_statistic": geom.statistics_face_attribute("area"),
            "cell_volume_statistic": geom.statistics_cell_attribute("volume"),
        }
        self._log.info(json.dumps(mesh_info))

    def on_task_begin(self, *args, **kwargs):
        self._log.info("Task begin.")

    def on_task_end(self, *args, **kwargs):
        # if self._perf_process is not None:
        #     self._perf_process.send_signal(subprocess.signal.SIGINT)
        #     self._perf_process.wait()
        #     self._log.info("Stop performance monitoring process")
        self._log.info("Task end.")

    def on_step_begin(self, *args, **kwargs):
        pass

    def on_step(self, solver_status: dict, *args, **kwargs):
        self._step += 1
        if self._step % self._interval != 0:
            return

        status = self._collect_running_info(solver_status)
        self._log.info(json.dumps(status))

    def on_step_end(self, *args, **kwargs):
        pass

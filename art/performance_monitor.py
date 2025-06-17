"""
Performance monitoring utilities for ART benchmarking and testing.
"""

import os
import threading
import time
from datetime import datetime
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
import psutil
from matplotlib import pyplot as plt

HAS_TENSORFLOW = find_spec("tensorflow") is not None
HAS_TORCH = find_spec("torch") is not None

# GPU monitoring using NVIDIA NVML
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        NVMLError,
    )

    nvmlInit()
    HAS_NVML = True
    GPU_COUNT = nvmlDeviceGetCount()
except (ImportError, NVMLError):
    HAS_NVML = False
    GPU_COUNT = 0


class ResourceMonitor:
    """
    Monitor system resources including CPU, memory, and GPU (if available).
    """

    def __init__(self, interval: float = 0.1) -> None:
        """
        Initialize the resource monitor.

        :param interval: Sampling interval in seconds.
        """
        self.interval = interval
        self.cpu_percentages: list[float] = []
        self.memory_usages: list[float] = []
        self.timestamps: list[float] = []
        self.gpu_usages: list[Any] = []
        self.gpu_memories: list[Any] = []
        self.stop_flag = False
        self.process = psutil.Process(os.getpid())

        # Check for GPU availability
        self.has_gpu = HAS_NVML and GPU_COUNT > 0

        self.data_lock = threading.Lock()

    def start(self) -> None:
        """Start monitoring resources in a background thread."""
        self.stop_flag = False
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self) -> None:
        """Stop the resource monitoring thread."""
        self.stop_flag = True
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=2.0)

    def _monitor_resources(self) -> None:
        """Resource monitoring loop that runs in a background thread."""
        while not self.stop_flag:
            with self.data_lock:
                # CPU usage (percent)
                cpu_percent = self.process.cpu_percent()
                self.cpu_percentages.append(cpu_percent)

                # Memory usage (MB)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_usages.append(memory_mb)

                # Timestamp
                self.timestamps.append(time.time())

                # GPUs
                if self.has_gpu:
                    usages = []
                    memories = []
                    for i in range(GPU_COUNT):
                        handle = nvmlDeviceGetHandleByIndex(i)
                        util = nvmlDeviceGetUtilizationRates(handle)
                        mem_info = nvmlDeviceGetMemoryInfo(handle)
                        usages.append(util.gpu)
                        # use used memory in MB
                        memories.append(mem_info.used / (1024**2))
                    self.gpu_usages.append(usages)
                    self.gpu_memories.append(memories)

                time.sleep(self.interval)

    def get_data(self) -> dict[str, list[float]]:
        """
        Get the collected monitoring data.

        :return: Dictionary containing resource usage time series
        """
        with self.data_lock:
            timestamps = self.timestamps.copy()
            cpu_percentages = self.cpu_percentages.copy()
            memory_usages = self.memory_usages.copy()

            gpu_usages = None
            gpu_memories = None
            if self.has_gpu:
                gpu_usages = self.gpu_usages.copy()
                gpu_memories = self.gpu_memories.copy()

        min_length = min(len(timestamps), len(cpu_percentages), len(memory_usages))

        if self.has_gpu and gpu_usages is not None and gpu_memories is not None:
            min_length = min(min_length, len(gpu_usages), len(gpu_memories))

        timestamps = [t - timestamps[0] for t in timestamps[:min_length]]
        cpu_percentages = cpu_percentages[:min_length]
        memory_usages = memory_usages[:min_length]

        data = {
            "time": timestamps,
            "cpu_percent": cpu_percentages,
            "memory_mb": memory_usages,
        }

        if self.has_gpu and gpu_usages is not None and gpu_memories is not None:
            data["gpu_percent"] = gpu_usages[:min_length]
            data["gpu_memory_mb"] = gpu_memories[:min_length]

        return data

    def get_summary(self) -> dict[str, float]:
        """
        Get summary statistics of resource usage.

        :return: Dictionary with min, max, mean values for each resource
        """
        data = self.get_data()
        duration = data["time"][-1] if data["time"] else 0

        summary = {
            "duration_seconds": duration,
            "cpu_percent_mean": np.mean(data["cpu_percent"]) if data["cpu_percent"] else 0,
            "cpu_percent_max": np.max(data["cpu_percent"]) if data["cpu_percent"] else 0,
            "memory_mb_mean": np.mean(data["memory_mb"]) if data["memory_mb"] else 0,
            "memory_mb_max": np.max(data["memory_mb"]) if data["memory_mb"] else 0,
        }

        if self.has_gpu:
            # flatten across samples and GPUs
            all_gpu = np.array(self.gpu_usages)
            all_mem = np.array(self.gpu_memories)
            summary.update(
                {
                    "gpu_percent_mean": float(np.mean(all_gpu)),
                    "gpu_percent_max": float(np.max(all_gpu)),
                    "gpu_memory_mb_mean": float(np.mean(all_mem)),
                    "gpu_memory_mb_max": float(np.max(all_mem)),
                }
            )
        return summary

    def plot_results(self, title: str | None = None) -> Any | None:
        """
        Generate plots of resource usage over time.

        :param title: Title for the plot
        :return: Figure object if matplotlib is available, None otherwise
        """
        data = self.get_data()

        n_plots = 2 + (2 if self.has_gpu else 0)
        fig, axes = plt.subplots(n_plots // 2, 2, figsize=(15, 10))
        fig.suptitle(title or "Resource Usage During Execution", fontsize=16)

        # Flatten axes array for easier indexing
        axes = axes.flatten()

        # CPU usage plot
        axes[0].plot(data["time"], data["cpu_percent"], "b-")
        axes[0].set_title("CPU Usage (%)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("CPU Usage (%)")
        axes[0].grid(True)

        # Memory usage plot
        axes[1].plot(data["time"], data["memory_mb"], "r-")
        axes[1].set_title("Memory Usage (MB)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Memory (MB)")
        axes[1].grid(True)

        # the axes length check is redundant here; has_gpu already validates this
        if self.has_gpu:
            # GPU usage plot
            axes[2].plot(data["time"], data["gpu_percent"], "g-")
            axes[2].set_title("GPU Usage (%)")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("GPU Usage (%)")
            axes[2].grid(True)

            # GPU memory plot
            axes[3].plot(data["time"], data["gpu_memory_mb"], "m-")
            axes[3].set_title("GPU Memory Usage (MB)")
            axes[3].set_xlabel("Time (s)")
            axes[3].set_ylabel("GPU Memory (MB)")
            axes[3].grid(True)

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        return fig


class PerformanceTimer:
    """
    Context manager for timing and monitoring the performance of a code block.
    """

    def __init__(self, task_name: str = "Task", plot: bool = False, save_data: bool = False):
        """
        Initialize the performance timer.

        :param task_name: Name of the task for reporting
        :param plot: Whether to generate and show a plot
        :param save_data: Whether to save the data
        """
        self.task_name = task_name
        self.plot = plot
        self.save_data = save_data
        self.monitor = ResourceMonitor()
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self) -> "PerformanceTimer":
        """Start monitoring when entering the context."""
        print(f"Starting performance measurement for: {self.task_name}")
        self.start_time = time.time()
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop monitoring when exiting the context."""
        self.end_time = time.time()
        self.monitor.stop()

        execution_time = self.end_time - self.start_time
        print(f"\nPerformance Summary for {self.task_name}:")
        print(f"Execution Time: {execution_time:.2f} seconds")

        summary = self.monitor.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value:.2f}")

        if self.plot:
            fig = self.monitor.plot_results(title=f"Resource Usage: {self.task_name}")
            if fig:
                plt.show()

        if self.save_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.task_name.replace(" ", "_").replace("/", "_")
            data_filename = f"performance_{safe_name}_{timestamp}.csv"

            df = pd.DataFrame(self.monitor.get_data())
            df.to_csv(data_filename, index=False)
            print(f"Performance data saved to {data_filename}")

    def get_data(self) -> dict[str, list[float]]:
        """Get the collected monitoring data."""
        return self.monitor.get_data()

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics of resource usage."""
        return self.monitor.get_summary()

"""
Performance monitoring utilities for ART benchmarking and testing.
"""
import time
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import pandas as pd
import psutil
import os
import numpy as np
from matplotlib import pyplot as plt

# GPU monitoring support
try:
    import gputil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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
        self.cpu_percentages: List[float] = []
        self.memory_usages: List[float] = []
        self.timestamps: List[float] = []
        self.gpu_usages: List[float] = []
        self.gpu_memories: List[float] = []
        self.stop_flag = False
        self.process = psutil.Process(os.getpid())

        # Check for GPU availability
        self.has_gpu = False
        if HAS_GPUTIL:
            self.has_gpu = len(gputil.getGPUs()) > 0
        elif HAS_TENSORFLOW:
            self.has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        elif HAS_TORCH:
            self.has_gpu = torch.cuda.is_available()

    def start(self) -> None:
        """Start monitoring resources in a background thread."""
        self.stop_flag = False
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self) -> None:
        """Stop the resource monitoring thread."""
        self.stop_flag = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)

    def _monitor_resources(self) -> None:
        """Resource monitoring loop that runs in a background thread."""
        while not self.stop_flag:
            # CPU usage (percent)
            cpu_percent = self.process.cpu_percent()
            self.cpu_percentages.append(cpu_percent)

            # Memory usage (MB)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_usages.append(memory_mb)

            # Timestamp
            self.timestamps.append(time.time())

            # GPU usage if available
            if self.has_gpu:
                try:
                    gpu_usage, gpu_memory = self._get_gpu_stats()
                    self.gpu_usages.append(gpu_usage)
                    self.gpu_memories.append(gpu_memory)
                except Exception:
                    self.gpu_usages.append(0)
                    self.gpu_memories.append(0)

            time.sleep(self.interval)

    def _get_gpu_stats(self) -> Tuple[float, float]:
        """
        Get GPU utilization and memory usage.

        :return: Tuple of (GPU utilization percentage, GPU memory usage in MB)
        """
        if HAS_GPUTIL:
            gpus = gputil.getGPUs()
            if gpus:
                return gpus[0].load * 100, gpus[0].memoryUsed
        elif HAS_TENSORFLOW:
            try:
                # TensorFlow doesn't directly expose GPU utilization
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                memory_mb = memory_info['current'] / (1024 * 1024)
                return 0, memory_mb
            except:
                pass
        elif HAS_TORCH and torch.cuda.is_available():
            # PyTorch doesn't directly expose GPU utilization
            memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            return 0, memory_mb

        return 0, 0

    def get_data(self) -> Dict[str, List[float]]:
        """
        Get the collected monitoring data.

        :return: Dictionary containing resource usage time series
        """
        relative_times = [t - self.timestamps[0] for t in self.timestamps] if self.timestamps else []
        data = {
            'time': relative_times,
            'cpu_percent': self.cpu_percentages,
            'memory_mb': self.memory_usages,
        }

        if self.has_gpu:
            data['gpu_percent'] = self.gpu_usages
            data['gpu_memory_mb'] = self.gpu_memories

        return data

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of resource usage.

        :return: Dictionary with min, max, mean values for each resource
        """
        data = self.get_data()
        duration = data['time'][-1] if data['time'] else 0

        summary = {
            'duration_seconds': duration,
            'cpu_percent_mean': np.mean(data['cpu_percent']) if data['cpu_percent'] else 0,
            'cpu_percent_max': np.max(data['cpu_percent']) if data['cpu_percent'] else 0,
            'memory_mb_mean': np.mean(data['memory_mb']) if data['memory_mb'] else 0,
            'memory_mb_max': np.max(data['memory_mb']) if data['memory_mb'] else 0,
        }

        if self.has_gpu and 'gpu_percent' in data:
            summary['gpu_percent_mean'] = np.mean(data['gpu_percent']) if data['gpu_percent'] else 0
            summary['gpu_percent_max'] = np.max(data['gpu_percent']) if data['gpu_percent'] else 0
            summary['gpu_memory_mb_mean'] = np.mean(data['gpu_memory_mb']) if data['gpu_memory_mb'] else 0
            summary['gpu_memory_mb_max'] = np.max(data['gpu_memory_mb']) if data['gpu_memory_mb'] else 0

        return summary

    def plot_results(self, title: Optional[str] = None) -> Optional[Any]:
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
        if n_plots > 2:
            axes = axes.flatten()
        else:
            axes = [axes]

        # CPU usage plot
        axes[0].plot(data['time'], data['cpu_percent'], 'b-')
        axes[0].set_title('CPU Usage (%)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].grid(True)

        # Memory usage plot
        axes[1].plot(data['time'], data['memory_mb'], 'r-')
        axes[1].set_title('Memory Usage (MB)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Memory (MB)')
        axes[1].grid(True)

        if self.has_gpu and len(axes) > 2:
            # GPU usage plot
            axes[2].plot(data['time'], data['gpu_percent'], 'g-')
            axes[2].set_title('GPU Usage (%)')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('GPU Usage (%)')
            axes[2].grid(True)

            # GPU memory plot
            axes[3].plot(data['time'], data['gpu_memory_mb'], 'm-')
            axes[3].set_title('GPU Memory Usage (MB)')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('GPU Memory (MB)')
            axes[3].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
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
        self.start_time = 0
        self.end_time = 0

    def __enter__(self) -> 'PerformanceTimer':
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
            safe_name = self.task_name.replace(' ', '_').replace('/', '_')
            data_filename = f"performance_{safe_name}_{timestamp}.csv"

            df = pd.DataFrame(self.monitor.get_data())
            df.to_csv(data_filename, index=False)
            print(f"Performance data saved to {data_filename}")

    def get_data(self) -> Dict[str, List[float]]:
        """Get the collected monitoring data."""
        return self.monitor.get_data()

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of resource usage."""
        return self.monitor.get_summary()

import unittest
import time
import numpy as np
from art.performance_monitor import ResourceMonitor, PerformanceTimer, HAS_TENSORFLOW, HAS_TORCH


class TestPerformanceMonitoring(unittest.TestCase):
    def test_resource_monitor_basic(self):
        """Test basic functionality of the resource monitor."""
        monitor = ResourceMonitor()
        monitor.start()

        # Generate some load
        _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)

        time.sleep(0.5)  # Allow some time for monitoring
        monitor.stop()

        data = monitor.get_data()
        summary = monitor.get_summary()

        # Basic integrity checks
        self.assertGreater(len(data["time"]), 0)
        self.assertGreater(len(data["cpu_percent"]), 0)
        self.assertGreater(len(data["memory_mb"]), 0)

        # Check summary stats
        self.assertGreater(summary["duration_seconds"], 0)
        self.assertGreaterEqual(summary["cpu_percent_max"], 0)
        self.assertGreaterEqual(summary["memory_mb_max"], 0)

    def test_performance_timer(self):
        """Test the performance timer context manager."""
        with PerformanceTimer("Test Task") as timer:
            # Generate some load
            _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
            time.sleep(0.5)

        # Check that data was collected
        data = timer.get_data()
        self.assertGreater(len(data["time"]), 0)
        self.assertGreater(len(data["cpu_percent"]), 0)


class TestGPUMonitoring(unittest.TestCase):
    def test_gpu_detection(self):
        """Test that GPU detection works correctly."""
        monitor = ResourceMonitor()
        # Check if has_gpu is correctly set based on available libraries
        self.assertEqual(monitor.has_gpu, (HAS_TENSORFLOW or HAS_TORCH))

    def test_gpu_data_collection(self):
        """Test GPU data is collected when available."""
        monitor = ResourceMonitor()
        monitor.start()

        tf = None
        torch = None

        # Create a workload that might use GPU if available
        if HAS_TENSORFLOW:
            import tensorflow as tf

            # Force TensorFlow to use GPU if available
            with tf.device("/GPU:0"):
                # Create and multiply large tensors to load the GPU
                a = tf.random.normal([5000, 5000])
                b = tf.random.normal([5000, 5000])
                c = tf.matmul(a, b)
                # Force execution
                result = c.numpy()
                self.assertIsNotNone(result)

        elif HAS_TORCH:
            import torch

            # Check if CUDA is available and create GPU tensor if so
            if torch.cuda.is_available():
                device = torch.device("cuda")
                # Create and multiply large tensors on GPU
                a = torch.randn(5000, 5000, device=device)
                b = torch.randn(5000, 5000, device=device)
                c = torch.matmul(a, b)
                # Force synchronization
                torch.cuda.synchronize()
                self.assertIsNotNone(c)

        # Allow monitoring for a moment
        time.sleep(2)
        monitor.stop()

        # Get the collected data
        data = monitor.get_data()
        summary = monitor.get_summary()

        # If we have GPU monitoring capability, verify data is collected
        if monitor.has_gpu:
            self.assertIn("gpu_percent", data)
            self.assertIn("gpu_memory_mb", data)
            self.assertGreater(len(data["gpu_percent"]), 0)
            self.assertGreater(len(data["gpu_memory_mb"]), 0)

            # Check summary contains GPU metrics
            self.assertIn("gpu_percent_max", summary)
            self.assertIn("gpu_memory_mb_max", summary)

            # If using TensorFlow or PyTorch with GPU, we expect some GPU usage
            if (HAS_TENSORFLOW and tf.config.list_physical_devices("GPU")) or (
                HAS_TORCH and torch.cuda.is_available()
            ):
                self.assertGreater(
                    summary["gpu_percent_max"],
                    0,
                    "GPU should show some usage when processing tensors",
                )

    def test_performance_timer_with_gpu(self):
        """Test the performance timer captures GPU metrics."""
        with PerformanceTimer("GPU Test", plot=True) as timer:
            # Similar workload as above
            if HAS_TENSORFLOW:
                import tensorflow as tf

                with tf.device("/GPU:0"):
                    a = tf.random.normal([5000, 5000])
                    b = tf.random.normal([5000, 5000])
                    c = tf.matmul(a, b)
                    result = c.numpy()
                    self.assertIsNotNone(result)  # not needed, but avoids false warnings
            elif HAS_TORCH:
                import torch

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    a = torch.randn(5000, 5000, device=device)
                    b = torch.randn(5000, 5000, device=device)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                    self.assertIsNotNone(c)  # not needed, but avoids false warnings

            time.sleep(1)

        # Verify timer captured GPU data if GPU is available
        data = timer.get_data()
        summary = timer.get_summary()

        if timer.monitor.has_gpu:
            self.assertIn("gpu_percent", data)
            self.assertIn("gpu_memory_mb", data)
            self.assertIn("gpu_percent_max", summary)

    def test_multiple_gpus(self):
        """Test monitoring with multiple GPUs if available."""
        # This test only runs if we have GPU monitoring and multiple GPUs
        monitor = ResourceMonitor()
        if not monitor.has_gpu:
            self.skipTest("No GPU monitoring available")

        # Check for multiple GPUs
        multi_gpu = False
        if HAS_TENSORFLOW:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            multi_gpu = len(gpus) > 1
        elif HAS_TORCH:
            import torch

            if torch.cuda.is_available():
                multi_gpu = torch.cuda.device_count() > 1

        if not multi_gpu:
            self.skipTest("Multiple GPUs not available")

        # Test with multiple GPUs
        monitor.start()

        # Create workload on multiple GPUs
        if HAS_TENSORFLOW:
            import tensorflow as tf

            # Run operations on different GPUs
            for i, _ in enumerate(gpus[:2]):  # Use first two GPUs
                with tf.device(f"/GPU:{i}"):
                    a = tf.random.normal([3000, 3000])
                    b = tf.random.normal([3000, 3000])
                    c = tf.matmul(a, b)
                    result = c.numpy()
                    self.assertIsNotNone(result)  # not needed, but avoids false warnings
        elif HAS_TORCH:
            import torch

            # Use first two GPUs
            for i in range(2):
                device = torch.device(f"cuda:{i}")
                a = torch.randn(3000, 3000, device=device)
                b = torch.randn(3000, 3000, device=device)
                c = torch.matmul(a, b)
                torch.cuda.synchronize(device)
                self.assertIsNotNone(c)  # not needed, but avoids false warnings

        time.sleep(2)
        monitor.stop()

        data = monitor.get_data()
        # For multiple GPUs, we should have GPU data as lists of lists
        # Each inner list represents data for one GPU
        self.assertIsInstance(
            data["gpu_percent"][0],
            list,
            "With multiple GPUs, data should be structured as lists of lists",
        )

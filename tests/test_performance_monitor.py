import unittest
import time
import numpy as np
from art.performance_monitor import ResourceMonitor, PerformanceTimer


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
        self.assertGreater(len(data['time']), 0)
        self.assertGreater(len(data['cpu_percent']), 0)
        self.assertGreater(len(data['memory_mb']), 0)

        # Check summary stats
        self.assertGreater(summary['duration_seconds'], 0)
        self.assertGreaterEqual(summary['cpu_percent_max'], 0)
        self.assertGreaterEqual(summary['memory_mb_max'], 0)

    def test_performance_timer(self):
        """Test the performance timer context manager."""
        with PerformanceTimer("Test Task") as timer:
            # Generate some load
            _ = np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)
            time.sleep(0.5)

        # Check that data was collected
        data = timer.get_data()
        self.assertGreater(len(data['time']), 0)
        self.assertGreater(len(data['cpu_percent']), 0)

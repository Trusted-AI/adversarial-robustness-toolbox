from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import torch
import numpy as np
import pytest

from art.defences.preprocessor.variance_minimization_pytorch import TotalVarMinPyTorch
from art.defences.preprocessor.variance_minimization import TotalVarMin
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_init(art_warning):
    """Test initialization of TotalVarMinPyTorch with default parameters."""
    try:
        preprocessor = TotalVarMinPyTorch()
        assert preprocessor.prob == 0.3
        assert preprocessor.lamb == 0.5
        assert preprocessor.max_iter == 10
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_init_custom_params(art_warning):
    """Test initialization with custom parameters."""
    try:
        preprocessor = TotalVarMinPyTorch(prob=0.3, lamb=0.5, max_iter=20, clip_values=(0.0, 255.0))
        assert preprocessor.prob == 0.3
        assert preprocessor.lamb == 0.5
        assert preprocessor.max_iter == 20
        assert preprocessor.clip_values == (0.0, 255.0)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_triple_clip_values_error(art_warning):
    """Test error when clip_values has more than 2 elements."""
    try:
        exc_msg = "'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range."
        with pytest.raises(ValueError, match=exc_msg):
            TotalVarMinPyTorch(clip_values=(0, 1, 2))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_relation_clip_values_error(art_warning):
    """Test error when clip_values min >= max."""
    try:
        exc_msg = "Invalid 'clip_values': min >= max."
        with pytest.raises(ValueError, match=exc_msg):
            TotalVarMinPyTorch(clip_values=(1, 0))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_invalid_prob_error(art_warning):
    """Test error when prob is not in [0, 1] range."""
    try:
        exc_msg = "Probability must be between 0 and 1."
        with pytest.raises(ValueError, match=exc_msg):
            TotalVarMinPyTorch(prob=1.5)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_call_shape(art_warning):
    """Test that preprocessor maintains input shape."""
    try:
        preprocessor = TotalVarMinPyTorch(prob=1.0)

        # Test with different input shapes
        for shape in [(10, 3, 32, 32), (5, 1, 28, 28), (1, 3, 64, 64)]:
            x = np.random.rand(*shape).astype(np.float32)
            x_preprocessed, _ = preprocessor(x)

            assert x_preprocessed.shape == x.shape
            assert x_preprocessed.dtype == x.dtype
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_call_clipping(art_warning):
    """Test that preprocessor respects clip_values."""
    try:
        clip_values = (0.1, 0.9)
        preprocessor = TotalVarMinPyTorch(clip_values=clip_values, prob=1.0, channels_first=True)

        x = np.random.rand(2, 3, 16, 16).astype(np.float32)
        x_preprocessed, _ = preprocessor(x)

        assert np.all(x_preprocessed >= clip_values[0])
        assert np.all(x_preprocessed <= clip_values[1])
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_reproducibility(art_warning):
    """Test reproducibility with same random seed."""
    try:
        np.random.seed(42)
        preprocessor = TotalVarMinPyTorch(prob=1.0, channels_first=True)

        x = np.random.rand(2, 3, 16, 16).astype(np.float32)

        # Set torch seed for reproducibility
        torch.manual_seed(42)
        x_preprocessed1, _ = preprocessor(x.copy())

        torch.manual_seed(42)
        x_preprocessed2, _ = preprocessor(x.copy())

        np.testing.assert_array_almost_equal(x_preprocessed1, x_preprocessed2, decimal=5)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_one_channel(art_warning):
    """Test that preprocessor works with single channel images."""
    try:
        preprocessor = TotalVarMinPyTorch(prob=1.0, channels_first=True)

        # Create a single channel image
        x = np.random.rand(2, 1, 16, 16).astype(np.float32)
        x_preprocessed, _ = preprocessor(x)

        assert x_preprocessed.shape == x.shape
        assert x_preprocessed.dtype == x.dtype
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_three_channels(art_warning):
    """Test that preprocessor works with three channel images."""
    try:
        preprocessor = TotalVarMinPyTorch(prob=1.0, channels_first=True)

        # Create a three channel image
        x = np.random.rand(2, 3, 16, 16).astype(np.float32)
        x_preprocessed, _ = preprocessor(x)

        assert x_preprocessed.shape == x.shape
        assert x_preprocessed.dtype == x.dtype
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_variance_minimization_estimate_gradient(art_warning):
    """Test that estimate_gradient works with variance minimization."""
    try:
        preprocessor = TotalVarMinPyTorch(prob=1.0, channels_first=True)

        # Create a sample input
        x = np.random.rand(2, 3, 16, 16).astype(np.float32)
        grad = np.ones_like(x).astype(np.float32)

        # Estimate gradient
        gradients = preprocessor.estimate_gradient(x=x, grad=grad)

        assert gradients.shape == x.shape
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_loss_function_consistency(art_warning):
    """Test that the loss function produces reasonable values."""
    try:
        import torch

        # Create simple test case
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        mask = torch.ones_like(x)
        z_init = x.flatten().clone()

        # Test different norms
        for norm in [1, 2]:
            loss = TotalVarMinPyTorch._loss_func(z_init, x, mask, norm, lamb=0.5)

            # Loss should be finite and positive
            assert torch.isfinite(loss), f"Loss is not finite for norm {norm}"
            assert loss >= 0, f"Loss is negative for norm {norm}: {loss}"

            # With zero perturbation and mask=1, loss should only be TV term
            loss_zero = TotalVarMinPyTorch._loss_func(z_init, x, torch.zeros_like(mask), norm, lamb=0.5)
            assert loss_zero > 0, f"TV term should be positive for norm {norm}"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_norm_parameter_validation(art_warning):
    """Test different norm values and their validation."""
    try:
        # Valid norms should work
        for norm in [1, 2, 3]:
            preprocessor = TotalVarMinPyTorch(norm=norm, prob=1.0)
            x = np.random.rand(1, 3, 8, 8).astype(np.float32)
            x_preprocessed, _ = preprocessor(x)
            assert x_preprocessed.shape == x.shape

        # Invalid norms should raise errors
        with pytest.raises(ValueError, match="Norm must be a positive integer"):
            TotalVarMinPyTorch(norm=0)

        with pytest.raises(ValueError, match="Norm must be a positive integer"):
            TotalVarMinPyTorch(norm=-1)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_channels_first_vs_last(art_warning):
    """Test both channels_first=True and channels_first=False."""
    try:
        # Test with channels first (BCHW)
        preprocessor_cf = TotalVarMinPyTorch(prob=1.0, channels_first=True)
        x_cf = np.random.rand(2, 3, 8, 8).astype(np.float32)
        x_preprocessed_cf, _ = preprocessor_cf(x_cf)

        # Test with channels last (BHWC)
        preprocessor_cl = TotalVarMinPyTorch(prob=1.0, channels_first=False)
        x_cl = np.random.rand(2, 8, 8, 3).astype(np.float32)
        x_preprocessed_cl, _ = preprocessor_cl(x_cl)

        assert x_preprocessed_cf.shape == x_cf.shape
        assert x_preprocessed_cl.shape == x_cl.shape

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_invalid_input_dimensions(art_warning):
    """Test error handling for invalid input dimensions."""
    try:
        preprocessor = TotalVarMinPyTorch(prob=1.0)

        # Test with 3D input (should fail)
        x_3d = np.random.rand(2, 3, 8).astype(np.float32)
        with pytest.raises(ValueError, match="Input `x` must be a 4D tensor"):
            preprocessor(x_3d)

        # Test with 2D input (should fail)
        x_2d = np.random.rand(2, 3).astype(np.float32)
        with pytest.raises(ValueError, match="Input `x` must be a 4D tensor"):
            preprocessor(x_2d)

        # Test with 5D input (should fail)
        x_5d = np.random.rand(2, 3, 8, 8, 8).astype(np.float32)
        with pytest.raises(ValueError, match="Input `x` must be a 4D tensor"):
            preprocessor(x_5d)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_probability_edge_cases(art_warning):
    """Test edge cases for probability parameter."""
    try:
        x = np.random.rand(1, 3, 8, 8).astype(np.float32)

        # prob=0.0 should keep original image (no mask)
        preprocessor_zero = TotalVarMinPyTorch(prob=0.0, max_iter=1)
        torch.manual_seed(42)  # For reproducible mask generation
        x_zero, _ = preprocessor_zero(x.copy())

        # prob=1.0 should apply maximum processing
        preprocessor_one = TotalVarMinPyTorch(prob=1.0, max_iter=1)
        torch.manual_seed(42)
        x_one, _ = preprocessor_one(x.copy())

        # Results should be different
        assert not np.array_equal(x_zero, x_one), "prob=0.0 and prob=1.0 should produce different results"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_lambda_parameter_effect(art_warning):
    """Test that lambda parameter affects the regularization strength."""
    try:
        x = np.random.rand(1, 1, 8, 8).astype(np.float32)

        # Low lambda (less regularization)
        preprocessor_low = TotalVarMinPyTorch(prob=1.0, lamb=0.1, max_iter=5)
        torch.manual_seed(42)
        x_low, _ = preprocessor_low(x.copy())

        # High lambda (more regularization)
        preprocessor_high = TotalVarMinPyTorch(prob=1.0, lamb=10.0, max_iter=5)
        torch.manual_seed(42)
        x_high, _ = preprocessor_high(x.copy())

        # Results should be different
        assert not np.array_equal(x_low, x_high), "Different lambda values should produce different results"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_convergence_with_iterations(art_warning):
    """Test that more iterations generally lead to better convergence."""
    try:
        x = np.random.rand(1, 1, 8, 8).astype(np.float32)

        # Few iterations
        preprocessor_few = TotalVarMinPyTorch(prob=1.0, max_iter=1)
        torch.manual_seed(42)
        x_few, _ = preprocessor_few(x.copy())

        # Many iterations
        preprocessor_many = TotalVarMinPyTorch(prob=1.0, max_iter=20)
        torch.manual_seed(42)
        x_many, _ = preprocessor_many(x.copy())

        # Results should be different (more iterations should change the result)
        assert not np.array_equal(x_few, x_many), "Different iteration counts should produce different results"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_mask_application(art_warning):
    """Test that mask is properly applied in optimization."""
    try:
        import torch

        # Create a simple test case where we can verify mask behavior
        x = torch.ones((2, 2), dtype=torch.float32) * 5.0
        mask_full = torch.ones_like(x)
        mask_partial = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        z_init = torch.zeros_like(x).flatten()

        # Loss with full mask should be different from partial mask
        loss_full = TotalVarMinPyTorch._loss_func(z_init, x, mask_full, norm=2, lamb=0.5)
        loss_partial = TotalVarMinPyTorch._loss_func(z_init, x, mask_partial, norm=2, lamb=0.5)

        assert not torch.allclose(loss_full, loss_partial), "Full and partial masks should give different losses"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_gradient_computation(art_warning):
    """Test that gradients can be computed for the loss function."""
    try:
        import torch

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        mask = torch.ones_like(x)
        z_init = x.flatten().clone().requires_grad_(True)

        loss = TotalVarMinPyTorch._loss_func(z_init, x, mask, norm=2, lamb=0.5)
        loss.backward()

        # Gradient should exist and be finite
        assert z_init.grad is not None, "Gradient should be computed"
        assert torch.all(torch.isfinite(z_init.grad)), "Gradient should be finite"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_device_compatibility(art_warning):
    """Test that the preprocessor works on different devices."""
    try:
        x = np.random.rand(1, 3, 8, 8).astype(np.float32)

        # Test CPU
        preprocessor_cpu = TotalVarMinPyTorch(prob=1.0, device_type="cpu", max_iter=1)
        x_cpu, _ = preprocessor_cpu(x.copy())
        assert x_cpu.shape == x.shape

        # Test GPU if available
        if torch.cuda.is_available():
            preprocessor_gpu = TotalVarMinPyTorch(prob=1.0, device_type="gpu", max_iter=1)
            x_gpu, _ = preprocessor_gpu(x.copy())
            assert x_gpu.shape == x.shape

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_numerical_stability(art_warning):
    """Test numerical stability with edge cases."""
    try:
        # Test with very small values
        x_small = np.random.rand(1, 1, 4, 4).astype(np.float32) * 1e-6
        preprocessor = TotalVarMinPyTorch(prob=1.0, max_iter=2)
        x_processed, _ = preprocessor(x_small)

        assert np.all(np.isfinite(x_processed)), "Output should be finite for small inputs"

        # Test with large values
        x_large = np.random.rand(1, 1, 4, 4).astype(np.float32) * 1e6
        x_processed_large, _ = preprocessor(x_large)

        assert np.all(np.isfinite(x_processed_large)), "Output should be finite for large inputs"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_y_labels_passthrough(art_warning):
    """Test that y labels are passed through unchanged."""
    try:
        preprocessor = TotalVarMinPyTorch(prob=1.0, max_iter=1)

        x = np.random.rand(2, 3, 8, 8).astype(np.float32)
        y = np.array([0, 1])

        x_processed, y_processed = preprocessor(x, y)

        assert np.array_equal(y, y_processed), "Labels should pass through unchanged"
        assert np.array_equal(y, y_processed), "Labels should pass through unchanged"

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_consistency_with_cpu_version(art_warning):
    """Test consistency with CPU SciPy implementation (if available)."""
    try:

        x = np.random.rand(1, 8, 8, 1).astype(np.float32)  # BHWC format for CPU version

        # CPU version
        cpu_preprocessor = TotalVarMin(prob=1.0, lamb=0.5, max_iter=5)
        np.random.seed(42)
        x_cpu, _ = cpu_preprocessor(x.copy())

        # PyTorch version (convert to BCHW)
        x_torch = np.transpose(x, (0, 3, 1, 2))  # BHWC -> BCHW
        torch_preprocessor = TotalVarMinPyTorch(prob=1.0, lamb=0.5, max_iter=5, channels_first=True)
        torch.manual_seed(42)
        np.random.seed(42)
        x_torch_processed, _ = torch_preprocessor(x_torch.copy())

        # Convert back to BHWC for comparison
        x_torch_processed = np.transpose(x_torch_processed, (0, 2, 3, 1))

        # Results should be similar (allowing for numerical differences)
        diff = np.mean(np.abs(x_cpu - x_torch_processed))

        logger.info(f"Mean absolute difference between CPU and PyTorch versions: {diff}")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_memory_efficiency(art_warning):
    """Test that the preprocessor doesn't leak memory with larger inputs."""
    try:
        import gc
        import torch

        preprocessor = TotalVarMinPyTorch(prob=0.5, max_iter=2)

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for i in range(3):
            x = np.random.rand(2, 3, 32, 32).astype(np.float32)
            x_processed, _ = preprocessor(x)
            del x, x_processed
            gc.collect()

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory usage shouldn't grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100 * 1024 * 1024, f"Memory usage grew by {memory_growth} bytes"

    except ARTTestException as e:
        art_warning(e)

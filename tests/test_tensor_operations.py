"""
Tests for tensor operations module.
"""

import numpy as np
import torch
import pytest
from src.tensor_operations import (
    numpy_to_tensor,
    tensor_to_numpy,
    create_gaussian_kernel,
    apply_gaussian_convolution,
    compare_numpy_vs_torch
)


class TestNumpyTensorConversion:
    """Tests for numpy-tensor conversion."""
    
    def test_numpy_to_tensor(self):
        """Test conversion from numpy to tensor."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = numpy_to_tensor(arr, device='cpu')
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
    
    def test_tensor_to_numpy(self):
        """Test conversion from tensor to numpy."""
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        arr = tensor_to_numpy(tensor)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
    
    def test_roundtrip_conversion(self):
        """Test that numpy -> tensor -> numpy preserves values."""
        original = np.random.rand(10, 10).astype(np.float32)
        tensor = numpy_to_tensor(original, device='cpu')
        back = tensor_to_numpy(tensor)
        
        np.testing.assert_array_almost_equal(original, back)


class TestGaussianKernel:
    """Tests for Gaussian kernel creation."""
    
    def test_kernel_shape(self):
        """Test kernel has correct shape for conv2d."""
        kernel = create_gaussian_kernel(size=5, sigma=1.0, device='cpu')
        
        # shape should be (out_channels, in_channels, h, w)
        assert kernel.shape == (1, 1, 5, 5)
    
    def test_kernel_normalized(self):
        """Test kernel sums to 1 (normalized)."""
        kernel = create_gaussian_kernel(size=5, sigma=1.0, device='cpu')
        
        total = kernel.sum().item()
        np.testing.assert_almost_equal(total, 1.0, decimal=5)
    
    def test_kernel_symmetric(self):
        """Test kernel is symmetric."""
        kernel = create_gaussian_kernel(size=5, sigma=1.0, device='cpu')
        k = kernel.squeeze().numpy()
        
        # check horizontal symmetry
        np.testing.assert_array_almost_equal(k, np.fliplr(k))
        # check vertical symmetry
        np.testing.assert_array_almost_equal(k, np.flipud(k))


class TestGaussianConvolution:
    """Tests for Gaussian convolution operation."""
    
    def test_output_shape_preserved(self):
        """Test that convolution preserves input shape."""
        data = torch.rand(50, 50)
        smoothed = apply_gaussian_convolution(data, kernel_size=5)
        
        assert smoothed.shape == data.shape
    
    def test_smoothing_reduces_variance(self):
        """Test that smoothing reduces local variance."""
        # create noisy data
        data = torch.rand(100, 100) * 100
        smoothed = apply_gaussian_convolution(data, kernel_size=5, sigma=2.0)
        
        # variance should decrease
        original_var = data.var().item()
        smoothed_var = smoothed.var().item()
        
        assert smoothed_var < original_var


class TestPerformanceComparison:
    """Tests for NumPy vs PyTorch comparison."""
    
    def test_comparison_returns_dict(self):
        """Test that comparison returns expected keys."""
        data = np.random.rand(100, 100).astype(np.float32)
        results = compare_numpy_vs_torch(data, num_iterations=2)
        
        assert 'numpy_time' in results
        assert 'torch_time' in results
        assert 'speedup' in results
        assert 'device' in results
    
    def test_positive_times(self):
        """Test that timing values are positive."""
        data = np.random.rand(50, 50).astype(np.float32)
        results = compare_numpy_vs_torch(data, num_iterations=2)
        
        assert results['numpy_time'] > 0
        assert results['torch_time'] > 0
        assert results['speedup'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

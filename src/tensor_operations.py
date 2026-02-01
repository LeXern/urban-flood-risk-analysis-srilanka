"""
Tensor Operations Module

PyTorch-based spatial analysis operations with GPU awareness.
Includes convolution operations and performance comparison with NumPy.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, uniform_filter
import time
from typing import Union, Tuple, Dict


def numpy_to_tensor(
    array: np.ndarray,
    device: str = 'auto'
) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.
    
    Parameters
    ----------
    array : np.ndarray
        Input NumPy array
    device : str
        Device to use: 'auto', 'cpu', 'cuda', or 'mps'
        'auto' will detect available GPU
    
    Returns
    -------
    torch.Tensor
        PyTorch tensor on specified device
    
    Example
    -------
    >>> rainfall_np = np.random.rand(100, 100)
    >>> rainfall_tensor = numpy_to_tensor(rainfall_np, device='auto')
    >>> print(rainfall_tensor.device)
    """
    # convert to tensor
    if hasattr(array, 'values'):
        array = array.values
        
    tensor = torch.tensor(array, dtype=torch.float32)
    
    # select device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # for Apple Silicon
        else:
            device = 'cpu'
    
    return tensor.to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor back to NumPy array.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input PyTorch tensor
    
    Returns
    -------
    np.ndarray
        NumPy array
    """
    return tensor.detach().cpu().numpy()


def create_gaussian_kernel(
    size: int = 5,
    sigma: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for convolution.
    
    Parameters
    ----------
    size : int
        Kernel size (must be odd)
    sigma : float
        Standard deviation of Gaussian
    device : str
        Device to create kernel on
    
    Returns
    -------
    torch.Tensor
        4D kernel tensor ready for conv2d (out_channels, in_channels, h, w)
    
    Example
    -------
    >>> kernel = create_gaussian_kernel(size=5, sigma=1.5)
    >>> # kernel shape: (1, 1, 5, 5)
    """
    # create 1D gaussian
    x = torch.arange(size, dtype=torch.float32, device=device)
    x = x - (size - 1) / 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    
    # create 2D kernel via outer product
    gauss_2d = torch.outer(gauss_1d, gauss_1d)
    gauss_2d = gauss_2d / gauss_2d.sum()  # normalize
    
    # reshape for conv2d: (out_channels, in_channels, h, w)
    kernel = gauss_2d.view(1, 1, size, size)
    
    return kernel


def apply_gaussian_convolution(
    data: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Apply Gaussian smoothing using 2D convolution.
    
    Parameters
    ----------
    data : torch.Tensor
        2D input data tensor
    kernel_size : int
        Size of Gaussian kernel
    sigma : float
        Standard deviation of Gaussian
    
    Returns
    -------
    torch.Tensor
        Smoothed data
    
    Example
    -------
    >>> rainfall_tensor = numpy_to_tensor(rainfall_data)
    >>> smoothed = apply_gaussian_convolution(rainfall_tensor, kernel_size=5)
    """
    device = data.device
    
    # ensure 4D shape for conv2d: (batch, channels, height, width)
    original_shape = data.shape
    if data.dim() == 2:
        data = data.unsqueeze(0).unsqueeze(0)
    elif data.dim() == 3:
        data = data.unsqueeze(1)
    
    # create kernel
    kernel = create_gaussian_kernel(kernel_size, sigma, device)
    
    # apply convolution with padding to maintain size
    padding = kernel_size // 2
    smoothed = F.conv2d(data, kernel, padding=padding)
    
    # restore original shape
    if len(original_shape) == 2:
        smoothed = smoothed.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        smoothed = smoothed.squeeze(1)
    
    return smoothed


def identify_storm_centers(
    rainfall: torch.Tensor,
    threshold_percentile: float = 90.0,
    min_region_size: int = 4
) -> torch.Tensor:
    """
    Identify spatially coherent storm centers using convolution.
    
    Parameters
    ----------
    rainfall : torch.Tensor
        2D rainfall intensity tensor
    threshold_percentile : float
        Percentile threshold for storm identification
    min_region_size : int
        Minimum connected region size (in pixels)
    
    Returns
    -------
    torch.Tensor
        Binary mask of storm center locations
    
    Example
    -------
    >>> storm_centers = identify_storm_centers(rainfall_tensor, threshold_percentile=95)
    """
    # smooth to find coherent regions
    smoothed = apply_gaussian_convolution(rainfall, kernel_size=5, sigma=1.5)
    
    # calculate threshold
    threshold = torch.quantile(smoothed.flatten(), threshold_percentile / 100.0)
    
    # create binary mask
    storm_mask = smoothed > threshold
    
    return storm_mask


def compare_numpy_vs_torch(
    data: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Compare performance of NumPy (scipy) vs PyTorch convolution.
    
    Parameters
    ----------
    data : np.ndarray
        2D input data for testing
    kernel_size : int
        Kernel size for convolution
    sigma : float
        Gaussian sigma
    num_iterations : int
        Number of iterations for timing
    
    Returns
    -------
    dict
        Dictionary with timing results
    
    Example
    -------
    >>> results = compare_numpy_vs_torch(rainfall_data, kernel_size=5)
    >>> print(f"NumPy: {results['numpy_time']:.4f}s")
    >>> print(f"PyTorch: {results['torch_time']:.4f}s")
    >>> print(f"Speedup: {results['speedup']:.2f}x")
    """
    results = {}
    
    # Handle xarray input
    if hasattr(data, 'values'):
        data = data.values
    
    # ========== NumPy (scipy) approach ==========
    numpy_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = gaussian_filter(data.astype(np.float32), sigma=sigma)
        numpy_times.append(time.perf_counter() - start)
    
    results['numpy_time'] = np.mean(numpy_times)
    results['numpy_std'] = np.std(numpy_times)
    
    # ========== PyTorch approach ==========
    # detect available device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    results['device'] = device
    
    # convert to tensor
    tensor_data = torch.tensor(data, dtype=torch.float32, device=device)
    
    # warmup for GPU
    _ = apply_gaussian_convolution(tensor_data, kernel_size, sigma)
    if device != 'cpu':
        torch.cuda.synchronize() if device == 'cuda' else None
    
    # time pytorch
    torch_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = apply_gaussian_convolution(tensor_data, kernel_size, sigma)
        if device == 'cuda':
            torch.cuda.synchronize()
        torch_times.append(time.perf_counter() - start)
    
    results['torch_time'] = np.mean(torch_times)
    results['torch_std'] = np.std(torch_times)
    
    # calculate speedup
    results['speedup'] = results['numpy_time'] / results['torch_time']
    
    return results


def batch_process_timesteps(
    rainfall_3d: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Process multiple timesteps efficiently using batch processing.
    
    Parameters
    ----------
    rainfall_3d : np.ndarray
        3D array (time, height, width)
    kernel_size : int
        Gaussian kernel size
    sigma : float
        Gaussian sigma
    
    Returns
    -------
    np.ndarray
        Smoothed 3D array
    
    Example
    -------
    >>> daily_rainfall = np.random.rand(365, 100, 100)  # 1 year daily
    >>> smoothed = batch_process_timesteps(daily_rainfall)
    """
    # detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # convert to tensor with batch dimension
    # shape: (time, height, width) -> (time, 1, height, width)
    tensor = torch.tensor(rainfall_3d, dtype=torch.float32, device=device)
    tensor = tensor.unsqueeze(1)  # add channel dimension
    
    # create kernel
    kernel = create_gaussian_kernel(kernel_size, sigma, device)
    
    # apply convolution to all timesteps at once
    padding = kernel_size // 2
    smoothed = F.conv2d(tensor, kernel, padding=padding)
    
    # convert back: (time, 1, height, width) -> (time, height, width)
    return tensor_to_numpy(smoothed.squeeze(1))


def print_gpu_info() -> None:
    """Print available GPU/device information."""
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA available: No")
    
    if torch.backends.mps.is_available():
        print("Apple MPS available: Yes (Apple Silicon GPU)")
    else:
        print("Apple MPS available: No")
    
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 50)


if __name__ == "__main__":
    print("Tensor operations module loaded successfully\n")
    print_gpu_info()
    
    # quick performance test
    print("\nRunning performance comparison...")
    test_data = np.random.rand(500, 500).astype(np.float32)
    results = compare_numpy_vs_torch(test_data, kernel_size=5)
    
    print(f"\nResults (500x500 array, 5x5 kernel):")
    print(f"  NumPy (scipy):  {results['numpy_time']*1000:.2f} ms")
    print(f"  PyTorch ({results['device']}): {results['torch_time']*1000:.2f} ms")
    print(f"  Speedup: {results['speedup']:.2f}x")

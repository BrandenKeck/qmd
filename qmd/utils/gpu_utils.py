"""
GPU acceleration utilities for QMD framework.

Provides GPU detection, memory management, and numpy/cupy abstraction layer.
"""

import os
import warnings
from typing import Optional, Union, Any
import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True

    # Check if GPU is actually available and basic CuPy operations work
    try:
        cp.cuda.Device(0).compute_capability
        # Test basic GPU operations that might fail with missing runtime libraries
        test_array = cp.array([1.0, 2.0, 3.0])
        result = cp.sum(test_array)
        # If we get here, GPU is working
        GPU_AVAILABLE = True
    except Exception as e:
        GPU_AVAILABLE = False
        HAS_CUPY = False
        if "CUDARuntimeError" in str(type(e)):
            warnings.warn("CuPy installed but no CUDA GPU detected")
        else:
            warnings.warn(f"CuPy installed but GPU operations failed ({e}). Falling back to CPU.")

    # Try to import optional CuPy submodules
    try:
        import cupyx.scipy.linalg as cp_linalg
    except ImportError:
        cp_linalg = None
        warnings.warn("cupyx.scipy.linalg not available - some advanced linear algebra features disabled")

    try:
        import cupyx.scipy.ndimage as cp_ndimage
    except ImportError:
        cp_ndimage = None
        warnings.warn("cupyx.scipy.ndimage not available - some image processing features disabled")

except ImportError:
    HAS_CUPY = False
    GPU_AVAILABLE = False
    # Create dummy cupy module for fallback
    cp = np
    cp_linalg = None
    cp_ndimage = None

# Global GPU configuration
_USE_GPU = GPU_AVAILABLE and os.environ.get('QMD_USE_GPU', 'true').lower() == 'true'
_GPU_DEVICE = int(os.environ.get('QMD_GPU_DEVICE', '0'))
_GPU_MEMORY_POOL = None

class GPUConfig:
    """Global GPU configuration manager."""

    @staticmethod
    def is_available() -> bool:
        """Check if GPU acceleration is available."""
        return GPU_AVAILABLE

    @staticmethod
    def is_enabled() -> bool:
        """Check if GPU acceleration is currently enabled."""
        return _USE_GPU and GPU_AVAILABLE

    @staticmethod
    def enable_gpu(device_id: int = 0) -> bool:
        """
        Enable GPU acceleration.

        Args:
            device_id: CUDA device ID to use

        Returns:
            True if GPU was successfully enabled, False otherwise
        """
        global _USE_GPU, _GPU_DEVICE

        if not GPU_AVAILABLE:
            warnings.warn("GPU acceleration not available")
            return False

        try:
            cp.cuda.Device(device_id).use()
            _USE_GPU = True
            _GPU_DEVICE = device_id
            print(f"GPU acceleration enabled on device {device_id}")
            return True
        except Exception as e:
            warnings.warn(f"Failed to enable GPU: {e}")
            _USE_GPU = False
            return False

    @staticmethod
    def disable_gpu() -> None:
        """Disable GPU acceleration, fallback to CPU."""
        global _USE_GPU
        _USE_GPU = False
        print("GPU acceleration disabled, using CPU")

    @staticmethod
    def get_device_info() -> dict:
        """Get information about available GPU devices."""
        if not HAS_CUPY:
            return {"available": False, "devices": []}

        devices = []
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            for i in range(device_count):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    memory_info = cp.cuda.runtime.memGetInfo()
                    devices.append({
                        "id": i,
                        "name": props["name"].decode(),
                        "compute_capability": f"{props['major']}.{props['minor']}",
                        "total_memory_gb": props["totalGlobalMem"] / (1024**3),
                        "free_memory_gb": memory_info[0] / (1024**3),
                        "used_memory_gb": memory_info[1] / (1024**3)
                    })
        except Exception as e:
            warnings.warn(f"Could not get device info: {e}")

        return {"available": len(devices) > 0, "devices": devices}

# Array abstraction layer
def get_array_module(arr: Optional[Union[np.ndarray, Any]] = None):
    """
    Get appropriate array module (numpy or cupy) based on current configuration.

    Args:
        arr: Optional array to check module for

    Returns:
        numpy or cupy module
    """
    if arr is not None and HAS_CUPY:
        try:
            return cp.get_array_module(arr)
        except Exception as e:
            warnings.warn(f"CuPy array module detection failed, using numpy: {e}")
            return np

    if GPUConfig.is_enabled():
        return cp
    else:
        return np

def ensure_array(arr: Union[np.ndarray, Any], target_device: str = 'auto') -> Union[np.ndarray, Any]:
    """
    Ensure array is on the correct device (CPU or GPU).

    Args:
        arr: Input array
        target_device: 'cpu', 'gpu', or 'auto' (follow global config)

    Returns:
        Array on correct device
    """
    if not HAS_CUPY:
        return np.asarray(arr)

    # Determine target device
    if target_device == 'auto':
        use_gpu = GPUConfig.is_enabled()
    elif target_device == 'gpu':
        use_gpu = GPU_AVAILABLE
    else:  # 'cpu'
        use_gpu = False

    # Convert array as needed
    if use_gpu:
        try:
            if isinstance(arr, np.ndarray):
                return cp.asarray(arr)
            else:
                return arr
        except Exception as e:
            # Fall back to CPU if GPU operations fail
            warnings.warn(f"GPU array conversion failed, falling back to CPU: {e}")
            return np.asarray(arr)
    else:
        if HAS_CUPY and isinstance(arr, cp.ndarray):
            try:
                return cp.asnumpy(arr)
            except Exception as e:
                warnings.warn(f"GPU to CPU conversion failed: {e}")
                return np.asarray(arr)
        else:
            return np.asarray(arr)

def to_cpu(arr: Union[np.ndarray, Any]) -> np.ndarray:
    """Convert array to CPU (numpy)."""
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        try:
            return cp.asnumpy(arr)
        except Exception as e:
            warnings.warn(f"GPU to CPU conversion failed: {e}")
            return np.asarray(arr)
    return np.asarray(arr)

def to_gpu(arr: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
    """Convert array to GPU (cupy) if available."""
    if HAS_CUPY and GPUConfig.is_available():
        try:
            return cp.asarray(arr)
        except Exception as e:
            warnings.warn(f"CPU to GPU conversion failed, returning CPU array: {e}")
            return np.asarray(arr)
    return arr

class MemoryPool:
    """GPU memory pool manager."""

    def __init__(self):
        self._pool = None
        if HAS_CUPY and GPUConfig.is_available():
            self._pool = cp.get_default_memory_pool()

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage."""
        if not self._pool:
            return {"available": False}

        return {
            "available": True,
            "used_bytes": self._pool.used_bytes(),
            "total_bytes": self._pool.total_bytes(),
            "used_gb": self._pool.used_bytes() / (1024**3),
            "total_gb": self._pool.total_bytes() / (1024**3)
        }

    def free_all_blocks(self):
        """Free all unused GPU memory blocks."""
        if self._pool:
            self._pool.free_all_blocks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_all_blocks()

# Initialize memory pool
memory_pool = MemoryPool()

def print_gpu_status():
    """Print current GPU configuration and status."""
    print("=== QMD GPU Status ===")
    print(f"CuPy available: {HAS_CUPY}")
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"GPU enabled: {GPUConfig.is_enabled()}")

    if GPUConfig.is_available():
        device_info = GPUConfig.get_device_info()
        for device in device_info["devices"]:
            print(f"Device {device['id']}: {device['name']} "
                  f"({device['free_memory_gb']:.1f}GB free)")

    memory_info = memory_pool.get_memory_info()
    if memory_info["available"]:
        print(f"GPU memory: {memory_info['used_gb']:.2f}GB used, "
              f"{memory_info['total_gb']:.2f}GB total")
    print("=" * 22)

# Environment variable configuration on import
if os.environ.get('QMD_GPU_VERBOSE', '').lower() == 'true':
    print_gpu_status()
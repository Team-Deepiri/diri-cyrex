"""
GPU Device Detection Utility
Automatically detects and uses GPU (CUDA) if available, falls back to CPU
"""
import os
import torch
from ..logging_config import get_logger

logger = get_logger("cyrex.device_detection")


def get_device() -> str:
    """
    Detect the best available device with proper fallback: CUDA → MPS → CPU
    
    Returns device string that can be used with PyTorch and SentenceTransformers.
    Actually tests GPU functionality, not just availability.
    """
    # Log diagnostic information for debugging
    logger.debug(f"PyTorch version: {torch.__version__}")
    logger.debug(f"CUDA available (torch.cuda.is_available()): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            # Get CUDA information
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_capability = torch.cuda.get_device_capability(0)
            
            logger.info(
                f"CUDA detected: version={cuda_version}, devices={device_count}, "
                f"GPU={gpu_name}, capability={cuda_capability[0]}.{cuda_capability[1]}"
            )
            
            # Check for RTX 5080/5090 (sm_120) compatibility issue
            if cuda_capability[0] >= 12:
                # Check if PyTorch supports this compute capability
                try:
                    # Try to get the list of supported compute capabilities
                    # PyTorch 2.9.1 with CUDA 12.6 supports up to sm_90
                    # RTX 5080/5090 requires sm_120 support (CUDA 12.8+)
                    test_tensor = torch.tensor([1.0], device='cuda')
                    result = test_tensor * 2.0
                    _ = result.cpu()
                    del test_tensor, result
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "no kernel image is available for execution on the device" in str(e) or \
                       "cudaErrorNoKernelImageForDevice" in str(e):
                        logger.error(
                            f"⚠️  RTX 5080/5090 (sm_{cuda_capability[0]}.{cuda_capability[1]}) detected, but PyTorch doesn't support this compute capability. "
                            f"Current PyTorch supports up to sm_90. "
                            f"To fix: Rebuild Docker image (CUDA 12.8 support should be automatic): "
                            f"docker-compose -f docker-compose.dev.yml build --no-cache cyrex"
                        )
                        raise
                    else:
                        raise
            
            # Test GPU functionality with a simple operation
            test_tensor = torch.tensor([1.0], device='cuda')
            result = test_tensor * 2.0
            _ = result.cpu()  # Ensure operation completes
            del test_tensor, result
            torch.cuda.empty_cache()
            
            logger.info(
                f"✅ CUDA GPU detected and tested successfully: {gpu_name} "
                f"(CUDA {cuda_version}, Capability {cuda_capability[0]}.{cuda_capability[1]})"
            )
            return "cuda"
        except RuntimeError as cuda_error:
            error_msg = str(cuda_error)
            # Check if this is the RTX 5080 compatibility issue
            if "no kernel image is available for execution on the device" in error_msg or \
               "cudaErrorNoKernelImageForDevice" in error_msg:
                logger.error(
                    f"❌ GPU compute capability not supported by current PyTorch installation. "
                    f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown'}, "
                    f"Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'Unknown'}. "
                    f"Error: {error_msg}. "
                    f"Solution: Rebuild Docker image (CUDA 12.8 support should be automatic): "
                    f"docker-compose -f docker-compose.dev.yml build --no-cache cyrex"
                )
            else:
                logger.warning(
                    f"CUDA available but GPU test failed: {error_msg}. "
                    f"Falling back to CPU. This may indicate: "
                    f"1) GPU not accessible in Docker container (check NVIDIA Container Toolkit), "
                    f"2) CUDA driver mismatch, or 3) GPU memory issue."
                )
        except Exception as cuda_error:
            logger.warning(
                f"CUDA available but test failed: {cuda_error}. Falling back to CPU"
            )
    else:
        # CUDA not available - provide diagnostic info
        logger.debug("CUDA not available via torch.cuda.is_available()")
        
        # Check if we're in Docker and might need NVIDIA Container Toolkit
        if os.path.exists("/.dockerenv"):
            logger.debug("Running in Docker container - ensure NVIDIA Container Toolkit is installed")
            # Check for NVIDIA runtime
            if os.path.exists("/proc/driver/nvidia"):
                logger.warning(
                    "NVIDIA driver detected in container but PyTorch CUDA not available. "
                    "This may indicate: 1) PyTorch not built with CUDA support, "
                    "2) CUDA libraries not in container, or 3) NVIDIA Container Toolkit not configured."
                )
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            result = test_tensor * 2.0
            _ = result.cpu()
            del test_tensor, result
            logger.info("✅ Apple Silicon (MPS) detected and tested successfully")
            return "mps"
        except Exception as mps_error:
            logger.warning(f"MPS available but test failed, falling back to CPU: {mps_error}")
    
    # Fallback to CPU
    logger.info("Using CPU device (no GPU detected or GPU test failed)")
    return "cpu"


def get_torch_device() -> torch.device:
    """Get PyTorch device object"""
    return torch.device(get_device())


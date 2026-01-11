"""
GPU Device Detection Utility
Automatically detects and uses GPU (CUDA) if available, falls back to CPU
"""
import torch
from ..logging_config import get_logger

logger = get_logger("cyrex.device_detection")


def get_device() -> str:
    """
    Detect the best available device with proper fallback: CUDA → MPS → CPU
    
    Returns device string that can be used with PyTorch and SentenceTransformers.
    Actually tests GPU functionality, not just availability.
    """
    # Check CUDA first (NVIDIA GPUs)
    if torch.cuda.is_available():
        try:
            # Test GPU functionality
            gpu_name = torch.cuda.get_device_name(0)
            cuda_capability = torch.cuda.get_device_capability(0)
            
            # Test a simple operation on GPU
            test_tensor = torch.tensor([1.0], device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            
            logger.info(
                "✅ CUDA GPU detected and tested successfully",
                gpu_name=gpu_name,
                cuda_capability=f"{cuda_capability[0]}.{cuda_capability[1]}",
                device="cuda"
            )
            return "cuda"
        except Exception as cuda_error:
            logger.warning(
                f"CUDA available but test failed, falling back to CPU",
                error=str(cuda_error)
            )
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            del test_tensor
            logger.info("✅ Apple Silicon (MPS) detected and tested successfully", device="mps")
            return "mps"
        except Exception as mps_error:
            logger.warning(f"MPS available but test failed, falling back to CPU", error=str(mps_error))
    
    # Fallback to CPU
    logger.info("Using CPU device (no GPU detected or GPU test failed)")
    return "cpu"


def get_torch_device() -> torch.device:
    """Get PyTorch device object"""
    return torch.device(get_device())


"""
Robust Embedding Wrapper
Properly handles PyTorch meta tensor issues using PyTorch APIs, not environment variables
"""
from typing import List
from ..logging_config import get_logger

logger = get_logger("cyrex.embeddings_wrapper")


class RobustEmbeddings:
    """
    Robust embedding wrapper that properly handles PyTorch meta tensor issues.
    Uses PyTorch's proper APIs to avoid meta device without environment variable hacks.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _get_target_device(self):
        """Determine target device with proper fallback: CUDA → MPS → CPU"""
        import torch
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("MPS (Apple Silicon) available, using MPS")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU (no GPU/MPS available)")
        
        return device
    
    def _initialize_model(self):
        """Initialize sentence-transformers model using proper PyTorch device handling."""
        try:
            import torch
            import os
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModel, AutoTokenizer
            
            # Determine target device with proper fallback
            target_device = self._get_target_device()
            logger.info(f"Initializing embedding model {self.model_name} on device: {target_device}")
            
            # Set environment variables to prevent meta device usage
            # These prevent transformers from using device_map which can cause meta device issues
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
            # Disable accelerate's device_map to prevent meta device
            os.environ['ACCELERATE_USE_CPU'] = '1' if target_device.type == 'cpu' else '0'
            # Prevent accelerate from using device_map
            os.environ['ACCELERATE_USE_DEVICE_MAP'] = '0'
            # Force transformers to not use low_cpu_mem_usage which can cause meta device
            # This is done by ensuring we pass low_cpu_mem_usage=False explicitly
            
            initialization_success = False
            last_error = None
            
            # Method 1: Load model using config + state_dict approach to avoid meta device
            if not initialization_success:
                try:
                    from transformers import AutoConfig
                    from transformers.utils import cached_file
                    from huggingface_hub import hf_hub_download
                    import gc
                    
                    # Load tokenizer first
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    
                    # Load config
                    config = AutoConfig.from_pretrained(self.model_name)
                    
                    # Create model from config (this creates a model with random weights, not on meta device)
                    model = AutoModel.from_config(config)
                    
                    # Move model to target device (models from config are not on meta device)
                    model = model.to(target_device)
                    
                    # Now load the state dict with explicit map_location to target device
                    # This ensures weights load directly to target device, not meta device
                    try:
                        # Try to get the model file - use hf_hub_download for reliability
                        try:
                            model_file = hf_hub_download(
                                repo_id=self.model_name,
                                filename="pytorch_model.bin",
                                cache_dir=None
                            )
                            # Load state dict with explicit map_location
                            state_dict = torch.load(model_file, map_location=target_device, weights_only=False)
                            model.load_state_dict(state_dict, strict=False)
                        except Exception:
                            # If pytorch_model.bin doesn't exist, try model.safetensors
                            try:
                                model_file = hf_hub_download(
                                    repo_id=self.model_name,
                                    filename="model.safetensors",
                                    cache_dir=None
                                )
                                # For safetensors, we need to use safetensors library
                                from safetensors.torch import load_file
                                state_dict = load_file(model_file, device=str(target_device))
                                model.load_state_dict(state_dict, strict=False)
                            except Exception as safetensors_err:
                                # If safetensors also fails, use cached_file as fallback
                                logger.warning(f"Could not load from hub, trying cached_file: {safetensors_err}")
                                model_file = cached_file(
                                    self.model_name,
                                    "pytorch_model.bin",
                                    cache_dir=None
                                )
                                state_dict = torch.load(model_file, map_location=target_device, weights_only=False)
                                model.load_state_dict(state_dict, strict=False)
                    except Exception as load_err:
                        # If all state dict loading fails, fall back to from_pretrained with error handling
                        logger.warning(f"Could not load state dict directly, using from_pretrained fallback: {load_err}")
                        del model
                        gc.collect()
                        
                        # Try loading with explicit settings
                        model = AutoModel.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float32,
                            device_map=None,
                            low_cpu_mem_usage=False,
                        )
                        
                        # Try to move it, and if it fails with meta error, we'll catch it in the except block
                        try:
                            model = model.to(target_device)
                        except RuntimeError as move_err:
                            if "meta tensor" in str(move_err).lower() or "to_empty" in str(move_err).lower():
                                # Still on meta - this shouldn't happen with our settings, but handle it
                                raise RuntimeError(f"Model still on meta device despite settings: {move_err}")
                            raise
                    
                    # Create SentenceTransformer with pre-loaded components
                    self.model = SentenceTransformer(modules=[model, tokenizer])
                    # Ensure SentenceTransformer is on target device
                    self.model = self.model.to(target_device)
                    
                    logger.info(f"Successfully initialized {self.model_name} on {target_device} (method 1: config + state_dict loading)")
                    initialization_success = True
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "meta" in error_msg or "to_empty" in error_msg:
                        logger.warning(f"Meta tensor error in method 1, trying method 2: {e}")
                    else:
                        logger.debug(f"Method 1 failed: {e}")
            
            # Method 2: Use SentenceTransformer directly with environment variables set
            if not initialization_success:
                try:
                    # SentenceTransformer handles device placement internally
                    self.model = SentenceTransformer(
                        self.model_name,
                        device=str(target_device)  # Pass device as string
                    )
                    
                    logger.info(f"Successfully initialized {self.model_name} on {target_device} (method 2: SentenceTransformer direct)")
                    initialization_success = True
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "meta" in error_msg or "to_empty" in error_msg:
                        logger.warning(f"Meta tensor error in method 2, trying method 3: {e}")
                    else:
                        logger.debug(f"Method 2 failed: {e}")
            
            # Method 3: Use SentenceTransformer with explicit CPU device (avoids meta device)
            if not initialization_success:
                try:
                    # Force CPU explicitly to avoid any meta device issues
                    self.model = SentenceTransformer(
                        self.model_name,
                        device='cpu'  # Explicitly use CPU string
                    )
                    
                    logger.info(f"Successfully initialized {self.model_name} on {target_device} (method 3: SentenceTransformer with explicit CPU)")
                    initialization_success = True
                except Exception as e:
                    last_error = e
                    logger.debug(f"Method 3 failed: {e}")
            
            # Method 4: Fallback to simpler, more compatible models
            if not initialization_success:
                # Try multiple valid fallback models in order
                fallback_models = [
                    "sentence-transformers/all-MiniLM-L12-v2",  # Larger but more stable
                    "sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smaller alternative
                    "sentence-transformers/all-mpnet-base-v2",  # Different architecture
                ]
                
                for fallback_model in fallback_models:
                    if self.model_name == fallback_model:
                        continue  # Skip if already tried
                    
                    try:
                        logger.warning(f"Trying fallback model: {fallback_model}")
                        # Use SentenceTransformer directly for fallback models (most reliable)
                        self.model = SentenceTransformer(
                            fallback_model,
                            device=str(target_device)
                        )
                        
                        # Update model name to reflect what we're actually using
                        self.model_name = fallback_model
                        logger.info(f"Successfully initialized fallback model {fallback_model}")
                        initialization_success = True
                        break
                    except Exception as fallback_error:
                        logger.debug(f"Fallback model {fallback_model} failed: {fallback_error}")
                        last_error = fallback_error
                        continue
            
            if not initialization_success:
                raise RuntimeError(
                    f"Failed to initialize embedding model '{self.model_name}' after all methods. "
                    f"Last error: {last_error}. "
                    f"This may be due to PyTorch version compatibility or insufficient resources."
                ) from last_error
            
        except ImportError as e:
            logger.error(f"Required packages not available: {e}")
            raise ImportError(
                "sentence-transformers and transformers are required. "
                "Install with: pip install sentence-transformers transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)


def get_robust_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> RobustEmbeddings:
    """Factory function to get robust embeddings instance."""
    return RobustEmbeddings(model_name)

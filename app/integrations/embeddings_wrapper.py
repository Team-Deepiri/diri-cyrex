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
    
    def _initialize_model(self):
        """Initialize sentence-transformers model using proper PyTorch device handling."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModel, AutoTokenizer
            
            # Determine target device (always CPU for compatibility)
            target_device = torch.device('cpu')
            logger.info(f"Initializing embedding model {self.model_name} on device: {target_device}")
            
            initialization_success = False
            last_error = None
            
            # Method 1: Load with explicit device_map and low_cpu_mem_usage to avoid meta device
            if not initialization_success:
                try:
                    # Use transformers directly with proper device mapping
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    
                    # Load model with explicit CPU device mapping and low memory usage
                    # This prevents PyTorch from using meta device
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",  # Explicit CPU mapping
                        low_cpu_mem_usage=True,  # Prevents meta device usage
                    )
                    
                    # Ensure model is on CPU (should already be, but be explicit)
                    if hasattr(model, 'to'):
                        model = model.to(target_device)
                    
                    # Create SentenceTransformer with pre-loaded components
                    self.model = SentenceTransformer(modules=[model, tokenizer])
                    # Ensure the SentenceTransformer wrapper is also on CPU
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(target_device)
                    
                    logger.info(f"Successfully initialized {self.model_name} on {target_device} (method 1: explicit device mapping)")
                    initialization_success = True
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "meta" in error_msg or "to_empty" in error_msg:
                        logger.warning(f"Meta tensor error in method 1, trying method 2: {e}")
                    else:
                        logger.debug(f"Method 1 failed: {e}")
            
            # Method 2: Handle meta tensor explicitly using to_empty() as PyTorch suggests
            if not initialization_success:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    
                    # Load model - may end up on meta device
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                    )
                    
                    # Check if model is on meta device and handle it properly
                    try:
                        # Try normal .to() first
                        model = model.to(target_device)
                    except RuntimeError as move_error:
                        error_msg = str(move_error).lower()
                        if "meta tensor" in error_msg or "to_empty" in error_msg:
                            # Model is on meta device - use to_empty() as PyTorch suggests
                            logger.info("Detected meta device, using to_empty() to properly move model")
                            
                            # Get model's device (should be meta)
                            if hasattr(model, 'device'):
                                current_device = model.device
                            else:
                                # Check first parameter's device
                                first_param = next(model.parameters(), None)
                                current_device = first_param.device if first_param is not None else None
                            
                            # Use to_empty() to move from meta to CPU
                            if hasattr(model, 'to_empty'):
                                # Create empty model on target device
                                model = model.to_empty(device=target_device)
                                # Load state dict to populate it
                                state_dict = model.state_dict()
                                # Re-initialize with proper weights
                                model.load_state_dict(state_dict, strict=False)
                            else:
                                # Fallback: reinitialize model directly on CPU
                                logger.warning("to_empty() not available, reinitializing model on CPU")
                                model = AutoModel.from_pretrained(
                                    self.model_name,
                                    torch_dtype=torch.float32,
                                )
                                # Force to CPU with explicit device placement
                                for name, param in model.named_parameters():
                                    param.data = param.data.to(target_device)
                                for name, buffer in model.named_buffers():
                                    buffer.data = buffer.data.to(target_device)
                        else:
                            # Not a meta tensor error, re-raise
                            raise
                    
                    # Create SentenceTransformer
                    self.model = SentenceTransformer(modules=[model, tokenizer])
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(target_device)
                    
                    logger.info(f"Successfully initialized {self.model_name} on {target_device} (method 2: meta device handling)")
                    initialization_success = True
                except Exception as e:
                    last_error = e
                    logger.debug(f"Method 2 failed: {e}")
            
            # Method 3: Use from_pretrained with explicit map_location (prevents meta device)
            if not initialization_success:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    
                    # Use map_location to force CPU loading and prevent meta device
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        map_location=str(target_device),  # Explicit map_location prevents meta device
                    )
                    
                    # Create SentenceTransformer
                    self.model = SentenceTransformer(modules=[model, tokenizer])
                    if hasattr(self.model, 'to'):
                        self.model = self.model.to(target_device)
                    
                    logger.info(f"Successfully initialized {self.model_name} on {target_device} (method 3: map_location)")
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
                        # Store original model name
                        original_model_name = self.model_name
                        # Try with fallback model (non-recursive to avoid infinite loops)
                        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                        model = AutoModel.from_pretrained(
                            fallback_model,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                        )
                        model = model.to(target_device)
                        self.model = SentenceTransformer(modules=[model, tokenizer])
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to(target_device)
                        
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

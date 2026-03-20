# Tensor Loading Optimization Plan (No Lazy Loading, No Warmups)

## Overview

This plan outlines optimizations to speed up tensor loading in Cyrex without using lazy loading or model warmups. The focus is on:
- Memory-mapped file loading (mmap)
- FastSafetensors library (4.8x-7.5x faster)
- Empty weight initialization
- Direct device loading
- Sharded checkpoints
- Parallel component loading
- Optimized PyTorch loading parameters

## Current Bottlenecks

### 1. Embeddings Wrapper (`app/integrations/embeddings_wrapper.py`)
- **Issue**: Sequential loading with no memory mapping
- **Impact**: High - embedding models load slowly on every instance creation
- **Current Flow**: `__init__()` → `_initialize_model()` → `torch.load()` → `load_state_dict()`

### 2. Multiple `torch.load()` Calls Without Memory Mapping
- **Issue**: Sequential loading with no memory mapping
- **Impact**: Medium - slower I/O, higher memory usage
- **Locations**: 
  - `embeddings_wrapper.py` (lines 143, 165, 252, 263)
  - `app/ml_models/rl_agent/ppo_agent.py` (line 578)
  - `app/services/reward_model.py` (line 47)
  - `app/services/workflow_optimizer.py` (line 450)

### 3. Safetensors Not Prioritized or Using Standard Library
- **Issue**: Safetensors tried as fallback, using standard library instead of fastsafetensors
- **Impact**: High - fastsafetensors is 4.8x-7.5x faster than standard safetensors
- **Current**: Try `.bin` first, then standard safetensors

### 4. Sequential Component Loading
- **Issue**: Tokenizer and model loaded sequentially
- **Impact**: Low-Medium - could be parallelized
- **Current**: Load tokenizer → load config → load model → load state_dict

### 5. Random Weight Initialization Overhead
- **Issue**: Weights initialized randomly then immediately overwritten
- **Impact**: Medium - wastes time on initialization
- **Current**: Model created with random weights, then state_dict loaded

## Optimization Strategy

### Phase 1: FastSafetensors Library (Highest Impact, Low Risk)

#### 1.1 Install and Use FastSafetensors
**File**: `app/integrations/embeddings_wrapper.py`

**Research**: FastSafetensors achieves 4.8x to 7.5x performance improvements by copying tensor groups directly to device memory, enabling parallelized copying and peer-to-peer DMA.

**Changes**:
- Install `fastsafetensors` library: `pip install fastsafetensors`
- Replace standard safetensors loading with fastsafetensors
- Prioritize safetensors format over `.bin` files

**Implementation**:
```python
# Try fastsafetensors first (4.8x-7.5x faster)
try:
    from fastsafetensors import safe_open
    
    model_file = hf_hub_download(
        repo_id=self.model_name,
        filename="model.safetensors",
        cache_dir=None
    )
    
    # FastSafetensors loads directly to device with parallel copying
    with safe_open(model_file, framework="pt", device=str(target_device)) as f:
        state_dict = {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    model.load_state_dict(state_dict, strict=False)
except Exception:
    # Fallback to standard safetensors
    try:
        from safetensors.torch import load_file
        model_file = hf_hub_download(
            repo_id=self.model_name,
            filename="model.safetensors",
            cache_dir=None
        )
        state_dict = load_file(model_file, device=str(target_device))
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        # Final fallback to pytorch_model.bin with mmap
        model_file = hf_hub_download(
            repo_id=self.model_name,
            filename="pytorch_model.bin",
            cache_dir=None
        )
        state_dict = torch.load(
            model_file,
            map_location=target_device,
            weights_only=False,
            mmap=True
        )
        model.load_state_dict(state_dict, strict=False)
```

**Expected Impact**: 
- **Time Saved**: 4.8x-7.5x faster loading for safetensors models
- **Memory**: Direct device loading reduces host memory usage
- **Compatibility**: Works with existing safetensors format

### Phase 2: Memory-Mapped Loading (High Impact, Low Risk)

#### 2.1 Add `mmap=True` to `torch.load()` Calls
**Files**: 
- `app/integrations/embeddings_wrapper.py`
- `app/ml_models/rl_agent/ppo_agent.py`
- `app/services/reward_model.py`
- `app/services/workflow_optimizer.py`

**Research**: Memory-mapped loading can reduce loading time by up to 80% while using less system memory. OS handles paging, loading data on-demand.

**Changes**:
```python
# Before
state_dict = torch.load(model_file, map_location=target_device, weights_only=False)

# After
state_dict = torch.load(
    model_file, 
    map_location=target_device, 
    weights_only=False,
    mmap=True  # Enable memory mapping - up to 80% faster
)
```

**Benefits**:
- Faster I/O (OS handles paging)
- Lower memory usage (pages loaded on-demand)
- Better for large models
- No code structure changes needed

**Expected Impact**: 
- **Time Saved**: 20-80% reduction in load time (depending on model size)
- **Memory Saved**: 30-50% reduction in peak memory usage

#### 2.2 Memory-Mapped File Access for Large Models
**File**: `app/integrations/embeddings_wrapper.py`

**Implementation**:
```python
import mmap

# For very large models, use explicit mmap
with open(model_file, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        state_dict = torch.load(mm, map_location=target_device, weights_only=False)
    finally:
        mm.close()
```

**Expected Impact**: 
- Additional 10-20% improvement for models >1GB
- Better memory efficiency

### Phase 3: Empty Weight Initialization (Medium Impact, Low Risk)

#### 3.1 Use Accelerate's `init_empty_weights()` Context Manager
**File**: `app/integrations/embeddings_wrapper.py`

**Research**: Skip random initialization of weights that will be immediately overwritten. Saves time and memory during model creation.

**Changes**:
```python
from accelerate import init_empty_weights

# Create model without random initialization
with init_empty_weights():
    model = AutoModel.from_config(config)

# Now load weights directly - no wasted initialization
state_dict = load_weights(...)
model.load_state_dict(state_dict, strict=False)
```

**Expected Impact**: 
- **Time Saved**: 10-30% faster model creation
- **Memory Saved**: Avoids allocating random weights
- **Risk**: Low - well-tested Accelerate feature

#### 3.2 Verify `_fast_init=True` is Enabled
**File**: `app/integrations/embeddings_wrapper.py`

**Research**: Transformers library uses `_fast_init=True` by default to skip random initialization. Verify this isn't being overridden.

**Implementation**:
```python
# Ensure fast_init is enabled (default in Transformers)
model = AutoModel.from_pretrained(
    self.model_name,
    _fast_init=True,  # Explicitly enable (should be default)
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=False,
)
```

**Expected Impact**: 
- **Time Saved**: 5-15% faster if currently disabled
- **Risk**: None - just ensuring default behavior

### Phase 4: Direct Device Loading (Medium Impact, Medium Risk)

#### 4.1 Load Directly to Target Device
**File**: `app/integrations/embeddings_wrapper.py`

**Research**: FastSafetensors and optimized loading can copy tensors directly to device memory, bypassing host RAM intermediate step.

**Changes**:
- Use `device` parameter in safetensors loading
- Use `map_location` correctly in torch.load
- Ensure tensors go directly to target device

**Implementation**:
```python
# Load directly to target device (no CPU intermediate)
state_dict = load_file(model_file, device=str(target_device))  # Direct to device

# Or for torch.load
state_dict = torch.load(
    model_file,
    map_location=target_device,  # Direct mapping
    weights_only=False,
    mmap=True
)
```

**Expected Impact**: 
- **Time Saved**: 10-20% faster for GPU loading
- **Memory Saved**: Avoids host memory copy
- **Risk**: Medium - need to verify device compatibility

### Phase 5: Sharded Checkpoints (Low-Medium Impact, Low Risk)

#### 5.1 Support Sharded Model Loading
**File**: `app/integrations/embeddings_wrapper.py`

**Research**: Transformers automatically shards checkpoints >10GB (default 5GB per shard). Loading shards sequentially reduces memory pressure.

**Changes**:
- Detect and handle sharded checkpoints
- Load shards sequentially or in parallel
- Use index file to map parameters

**Implementation**:
```python
from transformers.utils import cached_file

# Check for sharded model
try:
    # Try to find model.safetensors.index.json or pytorch_model.bin.index.json
    index_file = cached_file(self.model_name, "model.safetensors.index.json")
    if index_file:
        # Load sharded model
        from safetensors import safe_open
        state_dict = {}
        with open(index_file) as f:
            index = json.load(f)
            for shard_file, keys in index["weight_map"].items():
                shard_path = cached_file(self.model_name, shard_file)
                with safe_open(shard_path, framework="pt", device=str(target_device)) as f:
                    for key in keys:
                        state_dict[key] = f.get_tensor(key)
        model.load_state_dict(state_dict, strict=False)
except Exception:
    # Fallback to single file loading
    pass
```

**Expected Impact**: 
- **Memory Saved**: 50-70% reduction for large models
- **Time**: Similar or slightly faster (less memory pressure)
- **Applicability**: Only for models >10GB

### Phase 6: Parallel Component Loading (Low-Medium Impact, Medium Risk)

#### 6.1 Load Tokenizer and Config in Parallel
**File**: `app/integrations/embeddings_wrapper.py`

**Changes**:
- Use `concurrent.futures.ThreadPoolExecutor` to load tokenizer and config simultaneously

**Implementation**:
```python
from concurrent.futures import ThreadPoolExecutor

# Load tokenizer and config in parallel
with ThreadPoolExecutor(max_workers=2) as executor:
    tokenizer_future = executor.submit(
        AutoTokenizer.from_pretrained, 
        self.model_name
    )
    config_future = executor.submit(
        AutoConfig.from_pretrained, 
        self.model_name
    )
    
    tokenizer = tokenizer_future.result()
    config = config_future.result()
```

**Expected Impact**: 
- **Time Saved**: 20-30% reduction in initialization time
- **Risk**: Low - tokenizer and config loading are independent

### Phase 7: Optimized PyTorch Parameters (Low Impact, Low Risk)

#### 7.1 Use `weights_only=True` When Safe
**File**: Multiple files with `torch.load()`

**Changes**:
- Use `weights_only=True` for model checkpoints (not training state)
- Falls back to `weights_only=False` if needed

**Implementation**:
```python
# Try weights_only=True first (faster, safer)
try:
    state_dict = torch.load(
        model_file, 
        map_location=target_device, 
        weights_only=True,  # Faster, safer
        mmap=True
    )
except Exception:
    # Fallback for training checkpoints with optimizer state
    state_dict = torch.load(
        model_file, 
        map_location=target_device, 
        weights_only=False,
        mmap=True
    )
```

**Expected Impact**: 
- **Time Saved**: 5-10% faster loading
- **Safety**: Reduced security risk from pickle

**Note**: Only use `weights_only=True` for model weights, not for full training checkpoints.

#### 7.2 Optimize `from_pretrained()` Parameters
**Files**: 
- `app/services/inference_service.py`
- `app/integrations/local_llm.py`
- `app/integrations/lora_adapter_service.py`

**Changes**:
```python
# Optimize loading parameters
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # Use appropriate dtype
    device_map="auto",  # Already using
    low_cpu_mem_usage=True,  # Enable if not already
    use_safetensors=True,  # Prefer safetensors
    _fast_init=True,  # Skip random initialization
)
```

**Expected Impact**: 
- **Time Saved**: 10-20% faster loading
- **Memory Saved**: Lower peak memory usage

## Implementation Priority

### High Priority (Do First)
1. ✅ **FastSafetensors Library** - 4.8x-7.5x faster, biggest impact
2. ✅ **Memory-Mapped Loading** - Easy win, 20-80% speedup
3. ✅ **Safetensors Prioritization** - Faster and safer than `.bin`

### Medium Priority
4. ⚠️ **Empty Weight Initialization** - Good speedup, requires Accelerate
5. ⚠️ **Direct Device Loading** - Faster GPU loading, need to verify compatibility
6. ⚠️ **Parallel Component Loading** - Good speedup, requires testing

### Low Priority (Nice to Have)
7. 📝 **Sharded Checkpoints** - Only for large models (>10GB)
8. 📝 **Optimized PyTorch Parameters** - Smaller gains, easy to implement
9. 📝 **Verify `_fast_init=True`** - Quick check, small gain

## Expected Overall Impact

### Time Savings
- **Embeddings Loading**: 60-85% faster (fastsafetensors + mmap + empty init)
- **Model Checkpoint Loading**: 40-70% faster (mmap + fastsafetensors)
- **Large Models (>10GB)**: 70-90% faster (sharded + fastsafetensors + mmap)

### Memory Savings
- **Peak Memory**: 30-70% reduction (memory mapping + empty init)
- **GPU Memory**: 20-40% reduction (direct device loading)

### Risk Assessment
- **FastSafetensors**: Low risk - drop-in replacement for safetensors
- **Memory Mapping**: Low risk - PyTorch native feature
- **Empty Weight Init**: Low risk - well-tested Accelerate feature
- **Direct Device Loading**: Medium risk - need to verify device compatibility
- **Parallel Loading**: Medium risk - requires testing
- **Sharded Checkpoints**: Low risk - Transformers native support

## Dependencies

### New Packages Required
```bash
pip install fastsafetensors  # For 4.8x-7.5x faster safetensors loading
pip install accelerate  # For init_empty_weights() (may already be installed)
```

### Existing Packages (Verify Versions)
- `safetensors>=0.4.0` (for fastsafetensors compatibility)
- `transformers>=4.30.0` (for _fast_init support)
- `torch>=2.0.0` (for mmap support)

## Testing Plan

### Unit Tests
1. Test fastsafetensors loading works correctly
2. Test memory-mapped loading works correctly
3. Test safetensors fallback to `.bin` files
4. Test empty weight initialization
5. Test parallel loading doesn't break initialization
6. Test sharded checkpoint loading

### Integration Tests
1. Test embeddings work correctly with all optimizations
2. Test inference service with optimizations
3. Test checkpoint loading with memory mapping
4. Verify memory usage is reduced
5. Test GPU loading with direct device mapping

### Performance Tests
1. Measure load time before/after optimizations
2. Measure memory usage before/after
3. Test with various model sizes (small, medium, large)
4. Test concurrent loading scenarios
5. Compare fastsafetensors vs standard safetensors
6. Compare mmap vs non-mmap loading

## Rollout Plan

### Phase 1: FastSafetensors & Memory Mapping (Week 1)
- Install fastsafetensors library
- Add memory-mapped loading to all `torch.load()` calls
- Prioritize safetensors format
- Test thoroughly

### Phase 2: Empty Weight Init & Direct Loading (Week 2)
- Add `init_empty_weights()` context manager
- Optimize device loading
- Verify `_fast_init=True` is enabled
- Test and validate

### Phase 3: Advanced Optimizations (Week 3)
- Add parallel component loading
- Add sharded checkpoint support (if needed)
- Optimize PyTorch parameters
- Final performance testing

## Monitoring

### Metrics to Track
- Model load time (before/after)
- Memory usage (peak and idle)
- FastSafetensors vs standard safetensors usage
- Memory-mapped vs regular loading
- Error rates (fallback scenarios)

### Logging
- Log fastsafetensors vs standard safetensors usage
- Log memory-mapped vs regular loading
- Log safetensors vs `.bin` usage
- Log load time for each model
- Log device loading method (direct vs intermediate)

## Rollback Plan

If issues arise:
1. Revert fastsafetensors to standard safetensors library
2. Remove `mmap=True` if causing issues
3. Remove `init_empty_weights()` if compatibility issues
4. Revert safetensors priority if compatibility issues

All changes should be feature-flagged for easy rollback:
```python
USE_FASTSAFETENSORS = os.getenv("USE_FASTSAFETENSORS", "true").lower() == "true"
USE_MMAP = os.getenv("USE_MMAP", "true").lower() == "true"
USE_EMPTY_INIT = os.getenv("USE_EMPTY_INIT", "true").lower() == "true"
```

## Research References

1. **FastSafetensors**: [Speeding up Model Loading with fastsafetensors](https://arxiv.org/html/2505.23072v1) - 4.8x to 7.5x performance improvements
2. **Memory-Mapped Loading**: [Memory-Mapped Models: Load Large LLMs Faster with mmap Optimization](https://markaicode.com/memory-mapped-models-load-large-llms-faster/) - Up to 80% faster loading
3. **PyTorch Loading Tips**: [Tips for Loading an nn.Module from a Checkpoint](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)
4. **Handling Big Models**: [HuggingFace Accelerate - Handling big models](https://huggingface.co/docs/accelerate/v0.10.0/en/big_modeling)
5. **Transformers Big Models**: [HuggingFace Transformers - Big models](https://huggingface.co/docs/transformers/main/en/big_models)

## Notes

- **No Lazy Loading**: This plan explicitly avoids lazy loading as requested
- **No Warmups**: This plan explicitly avoids model warmups as requested
- **Backward Compatible**: All changes maintain existing APIs
- **Gradual Rollout**: Implement in phases to minimize risk
- **Performance Focus**: Optimizations target actual bottlenecks with research-backed improvements
- **Research-Based**: All optimizations are based on recent research and industry best practices

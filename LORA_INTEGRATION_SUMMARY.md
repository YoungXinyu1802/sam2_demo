# LIT-LoRA Integration with Demo Backend

## Overview
Integrated the LIT-LoRA functionality from `tools/online_eval.py` into the demo backend to support interactive LoRA-based mask correction.

## Key Changes to `demo/backend/server/inference/predictor.py`

### 1. Added LoRA Mode Management

#### New Instance Variables (lines 58-62):
```python
# LoRA mode flag
self.lit_lora_mode = False
# LoRA training data: {session_id: {obj_id: [gt_masks]}}
# Store GT masks for training, features are captured automatically by predictor
self.lora_training_data: Dict[str, Dict[int, List]] = {}
```

**Removed**: `self.lora_models` - No longer needed as predictor manages LoRA models internally

#### New Methods (lines 107-120):
```python
def enable_lora_mode(self):
    """Enable LIT-LoRA mode in the predictor"""
    if not self.lit_lora_mode:
        logger.info("Enabling LIT-LoRA mode")
        self.predictor.LIT_LoRA_mode = True
        self.predictor.reset_lora()
        self.lit_lora_mode = True
        
def disable_lora_mode(self):
    """Disable LIT-LoRA mode in the predictor"""
    if self.lit_lora_mode:
        logger.info("Disabling LIT-LoRA mode")
        self.predictor.LIT_LoRA_mode = False
        self.lit_lora_mode = False
```

### 2. Refactored `train_lora()` Method (lines 429-496)

**Before**: Manually extracted features (backbone, embeddings, etc.) and stored them as training samples

**After**: Simplified to use predictor's built-in feature capture

#### Key Changes:
1. **Auto-enable LoRA mode** when first training sample is added
2. **Trigger feature capture** by calling `predictor.add_new_mask()` which populates `temp_feat_for_lora`
3. **Store only GT masks** instead of full feature tuples
4. **Remove manual feature extraction** - predictor handles this automatically

#### New Workflow:
```python
# Enable LoRA mode if not already enabled
if not self.lit_lora_mode:
    self.enable_lora_mode()

# Add mask to trigger feature capture
input_mask = ((mask > 0) & (mask != 255)).astype(np.uint8)
self.predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=obj_id,
    mask=input_mask,
    run_mem_encoder=True
)

# Store only the GT mask
self.lora_training_data[session_id][obj_id].append({
    'frame_idx': frame_idx,
    'gt_mask': mask
})
```

### 3. Refactored `generate_lora_candidates()` Method (lines 498-616)

**Before**: Manually created and trained LoRA models using custom training loop

**After**: Uses predictor's built-in `train_lora()` and `lora_predict()` methods

#### Key Changes:
1. **Check if LoRA trained**: Uses `predictor.trained_lora` flag
2. **Train LoRA**: Calls `predictor.train_lora()` following online_eval.py pattern:
   ```python
   mask_decoder_lora = copy.deepcopy(self.predictor.sam_mask_decoder)
   self.predictor.convert_to_lora(mask_decoder_lora.transformer)
   self.predictor.freeze_non_lora(mask_decoder_lora)
   self.predictor.train_lora(mask_decoder_lora, gt_mask, training_epoch=100, mode='init')
   ```
3. **Generate predictions**: Uses `predictor.lora_predict()` instead of manual prediction
4. **Remove custom training logic**: No more manual optimizer, dataset loader, or training loop

#### New Workflow:
```python
# Train LoRA if not already trained
if not self.predictor.trained_lora:
    mask_decoder_lora = copy.deepcopy(self.predictor.sam_mask_decoder)
    self.predictor.convert_to_lora(mask_decoder_lora.transformer)
    self.predictor.freeze_non_lora(mask_decoder_lora)
    self.predictor.train_lora(mask_decoder_lora, gt_mask, training_epoch=100, mode='init')

# Generate predictions using LoRA
successful_predict, best_lora_iou, best_lora_idx, best_lora_predicted_mask_score = self.predictor.lora_predict(
    inference_state, 
    frame_idx, 
    dummy_gt_mask,
    correction_threshold=0.5
)
```

## How It Works

### Workflow Overview:
1. **Enable LoRA Mode**: First call to `train_lora()` automatically enables LIT_LoRA_mode
2. **Collect Training Samples**: Each correction call stores GT mask and triggers feature capture
3. **Train LoRA**: First call to `generate_lora_candidates()` trains initial LoRA model
4. **Generate Predictions**: Subsequent calls use `lora_predict()` to get best prediction from trained models

### Automatic Feature Capture:
When `LIT_LoRA_mode = True`, the predictor automatically captures features in `temp_feat_for_lora` during:
- `add_new_mask()` calls
- `propagate_in_video()` calls

These captured features include:
- `frame_idx`
- `pix_feat_with_mem` (memory-conditioned image features)
- `high_res_features` (multi-scale features)
- Image position encodings

### LoRA Training:
The predictor's `train_lora()` method:
1. Uses features from `temp_feat_for_lora`
2. Trains LoRA layers on transformer blocks
3. Stores trained models in `multi_lora` list
4. Uses replay buffer for fine-tuning

### LoRA Prediction:
The predictor's `lora_predict()` method:
1. Tests all trained LoRA models
2. Returns best prediction based on IoU
3. Supports multiple LoRA models (MoE approach)

## Benefits of This Approach

1. **Consistency**: Uses same code path as `online_eval.py`
2. **Simplicity**: Less manual feature extraction and management
3. **Correctness**: Predictor handles all dimension matching automatically
4. **Maintainability**: Single source of truth for LoRA logic
5. **Features**: Supports full LIT-LoRA capabilities (MoE, replay buffer, etc.)

## Integration with sam2_video_predictor.py

The predictor must have these methods (already implemented in your modified version):
- `reset_lora()` - Initialize LoRA state
- `convert_to_lora()` - Convert transformer to use LoRA layers
- `freeze_non_lora()` - Freeze non-LoRA parameters
- `train_lora()` - Train LoRA model on current features
- `lora_predict()` - Generate predictions using trained LoRA models

## Testing

To test the integration:
1. Start a session and propagate through video
2. Add corrections using `train_lora` endpoint
3. Request LoRA candidates using `generate_lora_candidates` endpoint
4. Check logs for "Enabling LIT-LoRA mode", "Training initial LoRA model", etc.

## Notes

- LoRA mode is enabled automatically on first training sample
- Features are captured automatically during propagation when LIT_LoRA_mode is enabled
- The predictor manages all LoRA models internally (stored in `multi_lora` list)
- Multiple LoRA models are supported for mixture-of-experts approach


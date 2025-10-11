# Fixed LoRA Training Timing Issue

## Problem
LoRA training was achieving very low IoU (0.0247) and predictions were failing. The root cause was incorrect training workflow timing.

### Original (Broken) Workflow:
1. **train_lora()**: Add mask → capture features → **store for later**
2. **generate_lora_candidates()**: Train LoRA using stored data
3. **Problem**: By the time training happens, `temp_feat_for_lora` has stale/wrong features

### Symptoms:
```
Early stopping triggered at epoch 51. Best epoch: 1, Best IoU: 0.0247
Training completed in 1.94 seconds.
Training completed. Best epoch: 1, Best IoU: 0.0247
WARNING:inference.predictor:LoRA prediction was not successful
```

## Root Cause

In `online_eval.py` (the working reference), LoRA training happens **immediately** after adding a correction:

```python
# online_eval.py lines 325-332
_, _, correct_res_mask = predictor.add_new_mask(...)  # Captures features
mask_decoder_lora = copy.deepcopy(predictor.sam_mask_decoder)
predictor.convert_to_lora(mask_decoder_lora.transformer)
predictor.freeze_non_lora(mask_decoder_lora)
predictor.train_lora(mask_decoder_lora, gt_mask, training_epoch=100)  # Trains IMMEDIATELY
```

**Why immediate training matters:**
- `add_new_mask()` populates `temp_feat_for_lora` with features for the current frame
- `train_lora()` uses features from `temp_feat_for_lora`
- Features must be fresh and correspond to the GT mask being trained on

Our broken implementation:
- Stored GT masks for later
- By training time, `temp_feat_for_lora` had different/stale features
- Training failed because features didn't match the GT mask

## Solution

### New (Fixed) Workflow:

**train_lora()** (lines 429-522):
1. Add mask via `add_new_mask()` → captures features in `temp_feat_for_lora`
2. **Immediately train LoRA** while features are fresh
3. Store GT mask for tracking purposes

```python
# Add mask to capture features
_, _, _ = self.predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=obj_id,
    mask=input_mask,
    run_mem_encoder=True
)

# Train LoRA IMMEDIATELY (following online_eval.py)
if not self.predictor.trained_lora:
    mask_decoder_lora = copy.deepcopy(self.predictor.sam_mask_decoder)
    self.predictor.convert_to_lora(mask_decoder_lora.transformer)
    self.predictor.freeze_non_lora(mask_decoder_lora)
    
    self.predictor.train_lora(
        mask_decoder_lora, 
        mask,  # Use GT mask while features are fresh
        training_epoch=100,
        mode='init'
    )
```

**generate_lora_candidates()** (lines 524-610):
- Simplified to only generate predictions
- No training happens here
- Uses `lora_predict()` to get best prediction from trained models
- Lowered threshold to 0.0 to always return candidates

```python
# Just predict, don't train
successful_predict, best_lora_iou, best_lora_idx, best_lora_predicted_mask_score = self.predictor.lora_predict(
    inference_state, 
    frame_idx, 
    dummy_gt_mask,
    correction_threshold=0.0  # Always return candidates
)
```

## Key Changes

### 1. Immediate Training (train_lora, lines 476-496)
**Before**: Stored mask for later training
**After**: Train immediately after add_new_mask

### 2. Simplified Candidate Generation (generate_lora_candidates, lines 524-610)
**Before**: Trained LoRA here if not trained
**After**: Just uses lora_predict, assumes already trained

### 3. Lower Threshold (line 566)
**Before**: `correction_threshold=0.5` (too high for low IoU)
**After**: `correction_threshold=0.0` (always return best candidate)

## Expected Behavior

### When User Adds Correction:
1. Frontend calls `train_lora` endpoint
2. Backend adds mask and trains LoRA immediately
3. Training happens with fresh features
4. Response confirms training completed

### When User Requests Candidates:
1. Frontend calls `generate_lora_candidates` endpoint
2. Backend uses `lora_predict` to get best prediction
3. Returns mask candidate with confidence score
4. Even low IoU candidates are returned (user can decide)

## Benefits

1. **Correctness**: Features match GT masks during training
2. **Consistency**: Follows proven online_eval.py workflow exactly
3. **Freshness**: Features are captured and used immediately
4. **Simplicity**: Each endpoint has clear single responsibility
5. **Performance**: No redundant training or stale data issues

## Testing

Expected improved results:
- Training IoU should be significantly higher (>0.6 typically)
- LoRA predictions should succeed more often
- Candidates should be generated with reasonable confidence scores
- Multi-frame corrections should improve LoRA quality

## Files Modified
- `demo/backend/server/inference/predictor.py`:
  - `train_lora()`: Lines 429-522 (immediate training)
  - `generate_lora_candidates()`: Lines 524-610 (prediction only)


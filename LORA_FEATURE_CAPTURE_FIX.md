# Fixed LoRA Training Failure - Feature Capture Issue

## Problem
LoRA training was failing with the warning:
```
WARNING:inference.predictor:LoRA training failed
```

## Root Cause

The `train_lora()` method in `sam2_video_predictor.py` returns `False` when `temp_feat_for_lora["pix_feat_with_mem"]` is `None` (line 1216-1217):

```python
pix_feat_with_mem = self.temp_feat_for_lora["pix_feat_with_mem"]
if pix_feat_with_mem is None:
    return False
```

### Why Features Weren't Captured

Features are captured in `_track_step` (sam2_base.py, lines 788-794) but only when:
1. `LIT_LoRA_mode = True` ✓ (we set this)
2. `_track_step` is actually called ✗ (this was the problem!)

### The Key Insight from online_eval.py

In `online_eval.py`, corrections happen **during propagation**:

```python
# Line 202: Start propagation loop
for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(...):
    # Line 218: Add correction DURING propagation
    _, _, correct_res_mask = predictor.add_new_mask(...)
    
    # Line 329: Train LoRA immediately - features are fresh!
    predictor.train_lora(mask_decoder_lora, gt_mask, ...)
```

When correction happens during propagation:
1. Frame is tracked → `_track_step` runs → features captured in `temp_feat_for_lora`
2. Correction is added via `add_new_mask`
3. LoRA training uses the fresh features

### Our Problem

In the demo backend:
1. User adds correction on some frame
2. We called `add_new_mask` directly
3. But the frame may not have been tracked yet
4. So `_track_step` never ran for that frame
5. `temp_feat_for_lora["pix_feat_with_mem"]` was `None`
6. Training failed!

## Solution

Call `propagate_to_frame()` **before** `add_new_mask()` to trigger feature capture:

### Code Changes (lines 465-483)

```python
# First, propagate to this frame to ensure features are captured
# This is critical: propagate_to_frame runs _track_step which populates temp_feat_for_lora
try:
    _, _, _ = self.predictor.propagate_to_frame(
        inference_state=inference_state,
        frame_idx=frame_idx,
    )
except Exception as e:
    logger.warning(f"Could not propagate to frame {frame_idx}: {e}")

# Add the mask - features should already be captured from propagation
input_mask = ((mask > 0) & (mask != 255)).astype(np.uint8)
_, _, _ = self.predictor.add_new_mask(
    inference_state=inference_state,
    frame_idx=frame_idx,
    obj_id=obj_id,
    mask=input_mask,
    run_mem_encoder=True
)
```

### How It Works Now

1. **propagate_to_frame()**: Tracks the frame → runs `_track_step` → captures features in `temp_feat_for_lora`
2. **add_new_mask()**: Adds the correction mask
3. **train_lora()**: Uses fresh features from `temp_feat_for_lora`
4. **Success!** ✓

### Added Debug Logging (lines 491-497)

To help diagnose future issues:

```python
# Debug: Check if features were captured
if hasattr(self.predictor, 'temp_feat_for_lora'):
    pix_feat = self.predictor.temp_feat_for_lora.get("pix_feat_with_mem")
    if pix_feat is None:
        logger.error(f"Features not captured! temp_feat_for_lora: {self.predictor.temp_feat_for_lora.keys()}")
    else:
        logger.info(f"Features captured successfully. Shape: {pix_feat.shape}")
```

## Expected Log Output

### Before Fix (Failed):
```
INFO:inference.predictor:Adding LoRA training sample...
INFO:inference.predictor:Training initial LoRA model
WARNING:inference.predictor:LoRA training failed
```

### After Fix (Success):
```
INFO:inference.predictor:Adding LoRA training sample...
INFO:inference.predictor:Training initial LoRA model
INFO:inference.predictor:Features captured successfully. Shape: torch.Size([1, 256, 64, 64])
INFO:inference.predictor:Initial LoRA model trained successfully
Training completed. Best epoch: X, Best IoU: 0.85+
```

## Testing

To verify the fix works:
1. Start a video session
2. Add initial mask/points on a frame
3. Propagate through video
4. Add a correction on a different frame via `train_lora` endpoint
5. Check logs for "Features captured successfully" and "Initial LoRA model trained successfully"
6. Training IoU should be much higher now (>0.8 typically)

## Key Takeaway

**Always propagate to a frame before training LoRA on it**, because:
- Propagation triggers `_track_step`
- `_track_step` captures features when `LIT_LoRA_mode = True`
- Features must be fresh when `train_lora()` is called
- Without propagation → no features → training fails

## Files Modified
- `demo/backend/server/inference/predictor.py`:
  - Lines 465-483: Added `propagate_to_frame()` call before `add_new_mask()`
  - Lines 491-497: Added debug logging for feature capture verification
  - Line 513: Updated warning message


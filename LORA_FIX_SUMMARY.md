# LoRA Training Dimension Mismatch Fix

## Problem
The LoRA training was failing with dimension mismatch errors:
1. First error: `The size of tensor a (256) must match the size of tensor b (320) at non-singleton dimension 3`
2. After partial fix: `The size of tensor a (256) must match the size of tensor b (64) at non-singleton dimension 3`

These occurred in the mask decoder when trying to add `src + dense_prompt_embeddings`.

## Root Causes
There were TWO issues in the demo backend's LoRA training data preparation:

### Issue 1: Wrong Mask Resize Target
**Problem**: Mask was being resized to video dimensions (e.g., 1920x1080) instead of the prompt encoder's expected input size.

**Why it matters**: The prompt encoder expects masks at `mask_input_size` (256x256 for base+ model), which is `4 * image_embedding_size`. It then downsamples them via `mask_downscaling` to match the image embedding resolution (64x64).

### Issue 2: Wrong FPN Level  
**Problem**: Using `backbone_fpn[0]` (first/coarsest level) instead of `backbone_fpn[-1]` (last/finest level) for image embeddings.

**Why it matters**: The SAM mask decoder expects image embeddings at `sam_image_embedding_size` (64x64 for base+ model). The FPN array is ordered from coarse to fine, and level 2 (the last level in default setting) is what SAM uses.

## Solution

Fixed both issues in `demo/backend/server/inference/predictor.py`:

### Fix 1: Correct FPN Level Selection (lines 460-463)
```python
# Before:
pix_feat_with_mem = backbone_out["backbone_fpn"][0]  # First level features
image_pe = backbone_out["vision_pos_enc"][0]  # Position encoding

# After:
# Use the finest level (last index) which matches what SAM decoder expects
pix_feat_with_mem = backbone_out["backbone_fpn"][-1]  # Finest level features
image_pe = backbone_out["vision_pos_enc"][-1]  # Position encoding for finest level
```

### Fix 2: Correct Mask Resizing (lines 472-478)
```python
# Before:
H = inference_state["video_height"]
W = inference_state["video_width"]
mask_resized = F.interpolate(
    mask_tensor,
    size=(H, W),
    mode="nearest"
)

# After:
# Resize to match the prompt encoder's expected mask input size
mask_resized = F.interpolate(
    mask_tensor,
    size=self.predictor.sam_prompt_encoder.mask_input_size,
    mode="bilinear",
    align_corners=False,
    antialias=True,
)
```

### Fix 3: Correct High-Res Features (line 490)
```python
# Before:
high_res_features = backbone_out["backbone_fpn"][1:]

# After:
# All levels except the finest one (which is used as main features)
high_res_features = backbone_out["backbone_fpn"][:-1] if len(backbone_out["backbone_fpn"]) > 1 else None
```

## How It Works Together

For base+ model (image_size=1024, backbone_stride=16):

1. **Image Embeddings** (`pix_feat_with_mem`):
   - From `backbone_fpn[-1]` (finest FPN level)
   - Shape: (B, 256, 64, 64)
   - 64x64 = 1024 / 16 = `image_size // backbone_stride`

2. **Mask Input**:
   - Resized to `mask_input_size` = (256, 256) 
   - Which is 4 * 64 = `4 * sam_image_embedding_size`
   - Shape: (B, 1, 256, 256)

3. **Dense Embeddings**:
   - Prompt encoder processes mask through `mask_downscaling`
   - Downsamples 256x256 → 64x64 (via two stride-2 conv layers)
   - Output shape: (B, 256, 64, 64)

4. **Dimension Match**: 
   - Image embeddings: (B, 256, 64, 64) ✓
   - Dense embeddings: (B, 256, 64, 64) ✓
   - Both match → addition succeeds in mask decoder!

## Why online_eval.py Works

`online_eval.py` uses the predictor's internal methods (`_forward_sam_heads`) which:
1. Automatically checks and resizes masks to `mask_input_size`
2. Uses the correct FPN level (finest level for SAM decoder)
3. The demo backend was manually extracting features and bypassing these checks

## Files Modified
- `demo/backend/server/inference/predictor.py`: 
  - Fixed FPN level selection (line 462)
  - Fixed mask resizing to use prompt encoder's mask_input_size (lines 472-478)
  - Fixed high-res features extraction (line 490)
  - Added proper GT mask resizing for training (lines 492-498)

## Testing
After applying these fixes:
1. Image embeddings and dense embeddings have matching spatial dimensions (64x64)
2. LoRA training proceeds without dimension mismatch errors
3. Behavior matches `online_eval.py` workflow

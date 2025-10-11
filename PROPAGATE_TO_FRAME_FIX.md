# Fixed Missing `propagate_to_frame` Method

## Problem
The demo backend was calling `predictor.propagate_to_frame()`, but this method was missing from the modified `sam2/sam2_video_predictor.py`, causing the error:

```
AttributeError: 'SAM2VideoPredictor' object has no attribute 'propagate_to_frame'
```

## Root Cause
During the modifications to add LoRA support to `sam2_video_predictor.py`, the `propagate_to_frame` method was accidentally removed. This method is used by the demo backend for frame-by-frame tracking controlled by video playback.

## Solution
Added the `propagate_to_frame` method back to `sam2/sam2_video_predictor.py` (lines 660-753).

### Method Implementation
The `propagate_to_frame` method:
1. Ensures outputs are consolidated via `propagate_in_video_preflight()`
2. Checks if the frame has already been tracked (returns cached result if so)
3. If not tracked, determines tracking direction (forward/backward) from nearest tracked frame
4. Runs single-frame inference for the requested frame
5. Returns frame index, object IDs, and video-resolution masks

### Key Features:
- **Caching**: Returns cached results for already-tracked frames (efficient)
- **Smart Tracking**: Automatically determines forward/backward tracking direction
- **Lazy Evaluation**: Only tracks requested frame, not entire video
- **Video Resolution**: Returns masks at original video resolution

## Usage in Demo Backend

The `propagate_to_frame` method is used in `demo/backend/server/inference/predictor.py` at line 410:

```python
def propagate_to_frame(self, request: PropagateToFrameRequest) -> PropagateDataResponse:
    """
    Propagate tracking to a specific frame. This is used for frame-by-frame tracking.
    """
    with self.autocast_context(), self.inference_lock:
        session_id = request.session_id
        frame_idx = request.frame_index
        
        session = self.__get_session(session_id)
        inference_state = session["state"]
        
        frame_idx, obj_ids, video_res_masks = self.predictor.propagate_to_frame(
            inference_state=inference_state,
            frame_idx=frame_idx,
        )
        
        masks_binary = (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
        # ... encode and return masks
```

## Workflow

### Frame-by-Frame Tracking Flow:
1. **User requests frame** → Backend calls `propagate_to_frame(inference_state, frame_idx)`
2. **Check cache** → If frame already tracked, return cached masks
3. **Track if needed** → Run inference only for this specific frame
4. **Return result** → Frame index, object IDs, and masks at video resolution

### Difference from `propagate_in_video`:
- `propagate_in_video`: Tracks all frames from start to end (full video)
- `propagate_to_frame`: Tracks single frame on-demand (interactive)

## Comparison with Original

The restored method is identical to the original implementation in `sam2_video_predictor_original.py` (lines 976-1069), ensuring compatibility with the demo backend and maintaining the expected behavior.

## Testing

To test the fix:
1. Start a video session in the demo
2. Add initial mask/points on a frame
3. Navigate to different frames using the video player
4. Each frame navigation triggers `propagate_to_frame`
5. Verify masks are displayed correctly on each frame

## Related Files Modified
- `sam2/sam2_video_predictor.py`: Added `propagate_to_frame` method (lines 660-753)

## Notes
- The method uses `@torch.inference_mode()` decorator for inference optimization
- It properly handles both conditioning frames (with user input) and non-conditioning frames
- Supports both forward and backward tracking from the nearest tracked frame
- Caching prevents redundant computation for previously tracked frames


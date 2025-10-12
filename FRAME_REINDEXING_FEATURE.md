# Frame Reindexing Feature for Frame-by-Frame Tracking

## Overview

This feature allows frame-by-frame tracking to use reindexed frame numbers (0, 1, 2, 3...) instead of the actual video frame indices (0, 12, 24, 36...) when tracking at a different FPS than the original video.

## Problem

When tracking at 5 FPS on a 60 FPS video:
- **Actual video frames**: 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120...
- **Tracking logic**: Works with large frame numbers that are harder to manage
- **User experience**: Confusing to work with non-sequential frame indices

## Solution

**Frame Reindexing**: Map reindexed frames to actual video frames for easier tracking logic.

### Example: 60 FPS Video, 5 FPS Tracking

| Reindexed Frame | Actual Video Frame | Calculation |
|----------------|-------------------|-------------|
| 0 | 0 | 0 * 12 = 0 |
| 1 | 12 | 1 * 12 = 12 |
| 2 | 24 | 2 * 12 = 24 |
| 3 | 36 | 3 * 12 = 36 |
| 4 | 48 | 4 * 12 = 48 |
| 5 | 60 | 5 * 12 = 60 |

**Formula**: `actual_frame_idx = reindexed_frame * (video_fps / tracking_fps)`

## Implementation

### 1. **Backend Changes**

#### Modified `propagate_to_frame()` Method
```python
def propagate_to_frame(self, inference_state, frame_idx, tracking_fps=None):
    # Calculate actual video frame index based on tracking FPS
    if tracking_fps is not None:
        video_fps = inference_state.get("video_fps", 60)
        frame_interval = int(video_fps / tracking_fps)
        actual_frame_idx = frame_idx * frame_interval
        print(f"Reindexed frame {frame_idx} -> actual video frame {actual_frame_idx}")
    else:
        actual_frame_idx = frame_idx
    
    # Use actual_frame_idx for all processing
    # ... rest of method uses actual_frame_idx
```

#### Updated Request Types
```python
@dataclass
class PropagateToFrameRequest(BaseRequest):
    type: str
    session_id: str
    frame_index: int
    tracking_fps: Optional[int] = None  # New parameter
```

#### Video FPS Detection
```python
def start_session(self, request: StartSessionRequest):
    # Get actual video FPS from metadata
    video_metadata = get_video_metadata(request.path)
    inference_state["video_fps"] = int(round(video_metadata.fps))
```

### 2. **Frontend Changes**

#### Updated Request Types
```typescript
export type TrackFrameRequest = Request<
  'trackFrame',
  {
    frameIndex: number;
    trackingFps?: number;  // New optional parameter
  }
>;
```

#### SAM2Model Integration
```typescript
public async trackFrame(frameIndex: number): Promise<void> {
  // Get tracking FPS from context
  const trackingFps = this._context.getTrackingFps();
  
  const requestBody = {
    session_id: sessionId,
    frame_index: frameIndex,
    tracking_fps: trackingFps,  // Pass tracking FPS
  };
}
```

#### VideoWorkerContext
```typescript
public getTrackingFps(): number {
  return this._trackingFps;
}
```

## Usage Examples

### Example 1: 60 FPS Video, 5 FPS Tracking
```typescript
// Frontend calls with reindexed frames
tracker.trackFrame(0);  // Tracks actual frame 0
tracker.trackFrame(1);  // Tracks actual frame 12
tracker.trackFrame(2);  // Tracks actual frame 24
tracker.trackFrame(3);  // Tracks actual frame 36
```

**Backend Logs:**
```
[SAM2VideoPredictor] Reindexed frame 0 -> actual video frame 0 (tracking at 5 FPS)
[SAM2VideoPredictor] Reindexed frame 1 -> actual video frame 12 (tracking at 5 FPS)
[SAM2VideoPredictor] Reindexed frame 2 -> actual video frame 24 (tracking at 5 FPS)
[SAM2VideoPredictor] Reindexed frame 3 -> actual video frame 36 (tracking at 5 FPS)
```

### Example 2: 30 FPS Video, 10 FPS Tracking
```typescript
// Frontend calls with reindexed frames
tracker.trackFrame(0);  // Tracks actual frame 0
tracker.trackFrame(1);  // Tracks actual frame 3
tracker.trackFrame(2);  // Tracks actual frame 6
tracker.trackFrame(3);  // Tracks actual frame 9
```

**Backend Logs:**
```
[SAM2VideoPredictor] Reindexed frame 0 -> actual video frame 0 (tracking at 10 FPS)
[SAM2VideoPredictor] Reindexed frame 1 -> actual video frame 3 (tracking at 10 FPS)
[SAM2VideoPredictor] Reindexed frame 2 -> actual video frame 6 (tracking at 10 FPS)
[SAM2VideoPredictor] Reindexed frame 3 -> actual video frame 9 (tracking at 10 FPS)
```

## Benefits

### 1. **Simplified Logic**
- Frontend works with sequential frame indices (0, 1, 2, 3...)
- No need to calculate actual frame numbers in frontend
- Easier debugging and development

### 2. **Better User Experience**
- Intuitive frame numbering for users
- Consistent with video playback controls
- Easier to understand frame relationships

### 3. **Flexible FPS Support**
- Works with any video FPS and tracking FPS combination
- Automatic calculation of frame intervals
- Backward compatible (no tracking_fps = no reindexing)

### 4. **Performance**
- No performance impact on tracking
- Same computational efficiency
- Maintains all existing optimizations

## Backward Compatibility

- **Optional Parameter**: `tracking_fps` is optional in all APIs
- **Default Behavior**: When `tracking_fps` is not provided, uses direct frame indexing
- **Existing Code**: Continues to work without changes
- **Gradual Adoption**: Can be enabled per session as needed

## Configuration

### Automatic Detection
- Video FPS is automatically detected from metadata
- Tracking FPS comes from user settings (FPS input box)
- Frame interval is calculated automatically

### Manual Override
```python
# Can still use direct frame indexing if needed
predictor.propagate_to_frame(inference_state, frame_idx=24)  # No reindexing
predictor.propagate_to_frame(inference_state, frame_idx=2, tracking_fps=5)  # With reindexing
```

This feature makes frame-by-frame tracking much more intuitive and easier to work with, especially when dealing with different FPS rates!

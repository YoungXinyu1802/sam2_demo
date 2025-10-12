# Frame Propagation Logging

## Overview

This feature adds comprehensive logging to track frame propagation throughout the entire pipeline, from frontend video playback to backend inference processing.

## Logging Points

### 1. Frontend - VideoWorkerContext
**Location**: `demo/frontend/src/common/components/video/VideoWorkerContext.ts`
**When**: When frame tracking is triggered during video playback
**Log Format**: 
```
[VideoWorkerContext] Triggering frame propagation for frame {frameIndex} (tracking at {trackingFps} FPS)
```

### 2. Frontend - SAM2Model Tracker
**Location**: `demo/frontend/src/common/tracker/SAM2Model.ts`
**When**: 
- Before making HTTP request to backend
- After receiving response from backend

**Log Formats**:
```
[SAM2Model] Starting frame propagation for frame {frameIndex} (session: {sessionId})
[SAM2Model] Frame propagation completed for frame {frameIndex}, received {maskCount} masks
```

### 3. Backend - InferenceAPI
**Location**: `demo/backend/server/inference/predictor.py`
**When**: 
- Before calling SAM2VideoPredictor
- After processing is complete

**Log Formats**:
```
[InferenceAPI] Starting frame propagation to frame {frameIndex} in session {sessionId}
[InferenceAPI] Frame propagation completed for frame {frameIndex}, processed {objectCount} objects, generated {maskCount} masks
```

### 4. Backend - SAM2VideoPredictor
**Location**: `sam2/sam2_video_predictor.py`
**When**: 
- When checking if frame is already cached
- When processing a new frame
- When returning results

**Log Formats**:
```
[SAM2VideoPredictor] Frame {frameIndex} already tracked, returning cached result
[SAM2VideoPredictor] Processing frame {frameIndex} for {objectCount} objects
[SAM2VideoPredictor] Frame {frameIndex} propagation completed, returning {objectCount} object masks
```

## Log Flow Example

When frame tracking is enabled at 5 FPS on a 30 FPS video, you'll see logs like this:

```
[VideoWorkerContext] Triggering frame propagation for frame 6 (tracking at 5 FPS)
[SAM2Model] Starting frame propagation for frame 6 (session: abc123)
[InferenceAPI] Starting frame propagation to frame 6 in session abc123
[SAM2VideoPredictor] Processing frame 6 for 2 objects
[SAM2VideoPredictor] Frame 6 propagation completed, returning 2 object masks
[InferenceAPI] Frame propagation completed for frame 6, processed 2 objects, generated 2 masks
[SAM2Model] Frame propagation completed for frame 6, received 2 masks

[VideoWorkerContext] Triggering frame propagation for frame 12 (tracking at 5 FPS)
[SAM2Model] Starting frame propagation for frame 12 (session: abc123)
[InferenceAPI] Starting frame propagation to frame 12 in session abc123
[SAM2VideoPredictor] Frame 12 already tracked, returning cached result
[SAM2VideoPredictor] Frame 12 propagation completed, returning 2 object masks
[InferenceAPI] Frame propagation completed for frame 12, processed 2 objects, generated 2 masks
[SAM2Model] Frame propagation completed for frame 12, received 2 masks
```

## Benefits

1. **Performance Monitoring**: Track which frames are being processed vs cached
2. **Debugging**: Identify bottlenecks in the frame propagation pipeline
3. **FPS Validation**: Verify that frame sampling is working at the expected rate
4. **Session Tracking**: Monitor frame propagation across different sessions
5. **Error Diagnosis**: Pinpoint where failures occur in the pipeline

## Usage

1. Open browser developer tools (F12)
2. Go to the Console tab
3. Enable frame tracking in the SAM2 demo
4. Watch the logs as frames are propagated during video playback

The logs will show:
- When frames are sampled based on the configured FPS
- Whether frames are processed fresh or returned from cache
- How many objects and masks are generated per frame
- The complete request/response flow through the system

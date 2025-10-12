# FPS Tracking Feature

## Overview

This feature adds a configurable FPS input box that allows users to set the frame rate for frame-by-frame tracking in the SAM2 demo application.

## Implementation Details

### Components Added/Modified

1. **New Atom**: `trackingFpsAtom` in `demo/frontend/src/demo/atoms.ts`
   - Manages the tracking FPS state (default: 5 FPS)

2. **VideoWorkerContext**: Added `setTrackingFps()` method
   - Allows runtime modification of the tracking FPS
   - Updates the `_trackingFps` private property
   - Affects frame sampling logic in `_isSampledFrame()`

3. **New Component**: `FPSInputBox` in `demo/frontend/src/common/components/input/FPSInputBox.tsx`
   - Number input with validation (1-60 FPS range)
   - Real-time updates to both state and video worker
   - Styled to match the application theme

4. **UI Integration**: Added to `ObjectsToolbarBottomActions.tsx`
   - Positioned next to the Frame Tracking button
   - Only visible when frame tracking controls are available

### Technical Flow

1. User changes FPS value in the input box
2. `FPSInputBox` component updates the `trackingFpsAtom` state
3. Component calls `video.setTrackingFps(newFps)` 
4. This triggers the VideoWorkerBridge to send a message to the VideoWorker
5. VideoWorker calls `context.setTrackingFps(fps)` on the VideoWorkerContext
6. The `_trackingFps` property is updated
7. Future frame sampling uses the new FPS value in `_isSampledFrame()`

### Frame Sampling Logic

The frame sampling is calculated as:
```typescript
const frameInterval = Math.round(videoFps / trackingFps);
return frameIndex % frameInterval === 0;
```

For example:
- Video at 30 FPS with tracking at 5 FPS = every 6th frame is sampled
- Video at 60 FPS with tracking at 10 FPS = every 6th frame is sampled
- Video at 24 FPS with tracking at 8 FPS = every 3rd frame is sampled

### Usage

1. Start a video session and add some points for tracking
2. The FPS input box will appear next to the "Track Frame-by-Frame" button
3. Adjust the FPS value (1-60) to control tracking frequency
4. Enable frame tracking to see the effect of the new FPS setting

### Benefits

- **Performance Control**: Lower FPS reduces computational load
- **Quality vs Speed**: Higher FPS provides more frequent updates but uses more resources
- **User Control**: Users can tune the tracking frequency based on their needs
- **Real-time Adjustment**: Changes take effect immediately without restarting the session

# Export Behavior Data on Final Frame - Implementation Summary

## Overview
Modified the video navigation system to automatically export behavior tracking data when the user navigates to the last reindexed/sampled frame using the right arrow key and there are no more frames to navigate to.

## Changes Made

### 1. VideoWorkerContext.ts
**Location:** `demo/frontend/src/common/components/video/VideoWorkerContext.ts`

**Changes:**
- Modified the `goToNextFrame()` method to detect when there are no more sampled frames to navigate to
- Added logic to emit `videoEndedAtFinalFrame` event when the user presses right arrow but there's no next frame
- This applies to the reindexed sampled frames (e.g., at 5 FPS sampling), not the total video frames

**Key Logic:**
```typescript
public goToNextFrame(): void {
  // ... calculate frame intervals and sampled frames
  
  if (nextSampledIndex < maxSampledFrames) {
    // Navigate to next sampled frame
    const nextFrame = Math.min(nextSampledIndex * frameInterval, this._decodedVideo.numFrames - 1);
    this.goToFrame(nextFrame);
  } else {
    // We've reached the last sampled frame - trigger export
    console.log('[VideoWorkerContext] Reached last sampled frame, triggering export');
    this.sendResponse('videoEndedAtFinalFrame');
  }
}
```

### 2. VideoWorkerTypes.ts
**Location:** `demo/frontend/src/common/components/video/VideoWorkerTypes.ts`

**Changes:**
- Added new response type: `VideoEndedAtFinalFrameResponse`
- Included the new response type in the `VideoWorkerResponse` union type

### 3. VideoWorkerBridge.ts
**Location:** `demo/frontend/src/common/components/video/VideoWorkerBridge.ts`

**Changes:**
- Added new event interface: `VideoEndedAtFinalFrameEvent`
- Added `videoEndedAtFinalFrame` to the `VideoWorkerEventMap` interface

### 4. DemoVideoEditor.tsx
**Location:** `demo/frontend/src/common/components/video/editor/DemoVideoEditor.tsx`

**Changes:**
- Added import for `VideoEndedAtFinalFrameEvent`
- Implemented event listener `onVideoEndedAtFinalFrame` that:
  - Calls `behaviorTracker.endSession()` to mark the session end time
  - Calls `behaviorTracker.downloadData()` to automatically download the behavior data
- Added proper cleanup in the useEffect return function

**Event Handler:**
```typescript
function onVideoEndedAtFinalFrame(_event: VideoEndedAtFinalFrameEvent) {
  // Automatically export behavior data when video reaches the final frame
  console.log('[DemoVideoEditor] Video reached final frame, exporting behavior data');
  behaviorTracker.endSession();
  behaviorTracker.downloadData();
}
```

## Behavior

### Before Changes:
- User navigates through sampled frames using right arrow key
- When at the last sampled frame, pressing right arrow does nothing
- User must manually click "Export Behavior Data" button to download tracking data

### After Changes:
- User navigates through sampled frames using right arrow key
- When at the last sampled frame, pressing right arrow triggers automatic export
- Behavior tracking data is automatically exported and downloaded as a JSON file
- Manual export button remains available for intermediate exports

## Understanding Sampled Frames

**Example:** If you have a 60 FPS video with 600 total frames, and frame tracking is set to 5 FPS:
- **Frame interval:** 60 / 5 = 12 frames
- **Sampled frames:** 0, 12, 24, 36, 48, 60, 72, ... 588
- **Total sampled frames:** ~50 frames
- **When right arrow is pressed at frame 588:** Export is automatically triggered

The export is triggered based on the **sampled frame count**, not the total video frame count. This is designed for the frame-by-frame tracking workflow where users navigate through reindexed frames.

## Testing Recommendations

1. **Basic Navigation Test:**
   - Load a video and enable frame tracking mode
   - Use right arrow key to navigate through sampled frames
   - Verify that at the last sampled frame, pressing right arrow triggers export

2. **Frame Tracking Mode Test:**
   - Enable frame tracking mode
   - Navigate through frames with object tracking
   - Verify behavior data contains all corrections

3. **Manual Export Test:**
   - Verify the manual "Export Behavior Data" button still works
   - Confirm users can export data before reaching the final sampled frame

4. **Edge Cases:**
   - Test with very short videos (< 10 frames)
   - Test different tracking FPS settings (5 FPS, 10 FPS, etc.)
   - Test navigating backwards and forwards

## Files Modified

1. `/demo/frontend/src/common/components/video/VideoWorkerContext.ts`
2. `/demo/frontend/src/common/components/video/VideoWorkerTypes.ts`
3. `/demo/frontend/src/common/components/video/VideoWorkerBridge.ts`
4. `/demo/frontend/src/common/components/video/editor/DemoVideoEditor.tsx`

## Backward Compatibility

- All existing functionality remains intact
- Manual export button continues to work
- No breaking changes to existing APIs
- New event is additive and doesn't affect other components
- Video playback (play button) behavior is unchanged

## Notes

- Console logging added for debugging: 
  - `[VideoWorkerContext] Reached last sampled frame, triggering export`
  - `[DemoVideoEditor] Video reached final frame, exporting behavior data`
- The behavior tracker data includes session timing, click events, tracking events, and frame-by-frame corrections
- Downloaded file format: `behavior-tracking-{timestamp}.json`
- This feature is specifically designed for frame-by-frame navigation workflow using arrow keys

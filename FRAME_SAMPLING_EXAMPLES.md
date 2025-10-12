# Frame Sampling Examples

## Updated Frame Sampling Logic

The frame sampling now uses a consistent interval-based approach starting from frame 0:

**Formula**: `interval = Math.round(videoFps / trackingFps)`
**Sampled Frames**: `0, interval, 2*interval, 3*interval, 4*interval, ...`

## Examples

### Example 1: 60 FPS Video, 5 FPS Tracking
- **Interval**: `60 / 5 = 12`
- **Sampled Frames**: `0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, ...`

### Example 2: 60 FPS Video, 10 FPS Tracking  
- **Interval**: `60 / 10 = 6`
- **Sampled Frames**: `0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, ...`

### Example 3: 30 FPS Video, 5 FPS Tracking
- **Interval**: `30 / 5 = 6`
- **Sampled Frames**: `0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, ...`

### Example 4: 24 FPS Video, 8 FPS Tracking
- **Interval**: `24 / 8 = 3`
- **Sampled Frames**: `0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, ...`

### Example 5: 120 FPS Video, 15 FPS Tracking
- **Interval**: `120 / 15 = 8`
- **Sampled Frames**: `0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, ...`

## Debugging

### Console Logs
When frame tracking is enabled, you'll see logs like:
```
[Frame Sampling] Video 60 FPS, Tracking 5 FPS, Interval 12, Sampled frame: 0
[Frame Sampling] Video 60 FPS, Tracking 5 FPS, Interval 12, Sampled frame: 12
[Frame Sampling] Video 60 FPS, Tracking 5 FPS, Interval 12, Sampled frame: 24
[Frame Sampling] Video 60 FPS, Tracking 5 FPS, Interval 12, Sampled frame: 36
```

### JavaScript API
You can also access the sampling information programmatically:

```javascript
// Get the frame sampling interval
const interval = video.getFrameSamplingInterval();
console.log(`Frame sampling interval: ${interval}`);

// Get the first 50 sampled frames
const sampledFrames = video.getSampledFrames(50);
console.log('Sampled frames:', sampledFrames);
// Output for 60 FPS video at 5 FPS: [0, 12, 24, 36, 48]
```

## Key Benefits

1. **Consistent Sampling**: Always starts from frame 0 and maintains regular intervals
2. **Predictable Pattern**: Easy to understand and debug
3. **Optimal Distribution**: Frames are evenly distributed across the video timeline
4. **Configurable**: Can adjust tracking FPS to change sampling density

## Implementation Details

The sampling logic is implemented in `VideoWorkerContext._isSampledFrame()`:

```typescript
private _isSampledFrame(frameIndex: number): boolean {
  if (!this._decodedVideo) return false;
  
  const videoFps = Math.round(this._decodedVideo.fps);
  const trackingFps = Math.round(this._trackingFps);
  
  // Calculate interval: original FPS / tracking FPS
  const frameInterval = Math.round(videoFps / trackingFps);
  
  // Always include frame 0, then sample at regular intervals
  return frameIndex % frameInterval === 0;
}
```

This ensures that:
- Frame 0 is always included
- Subsequent frames are sampled at regular intervals
- The interval is calculated as `videoFps / trackingFps`
- Rounding ensures integer frame indices

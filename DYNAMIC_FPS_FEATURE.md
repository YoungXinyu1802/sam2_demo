# Dynamic Video FPS Detection Feature

## Overview

This feature modifies the SAM2 demo application to automatically detect and use the actual FPS of uploaded videos for encoding and decoding, instead of using a hardcoded 24 FPS value.

## Problem

Previously, the application used a hardcoded `VIDEO_ENCODE_FPS=24` environment variable for all video processing, regardless of the actual FPS of the uploaded video. This could cause:

- **Quality Issues**: Videos with different FPS (30, 60, etc.) would be forced to 24 FPS, potentially causing frame rate conversion artifacts
- **Performance Issues**: Unnecessary frame rate conversion could slow down processing
- **Inconsistent Behavior**: All videos would be processed at the same FPS regardless of their original characteristics

## Solution

The system now:

1. **Detects Video FPS**: Automatically extracts the FPS from uploaded video metadata
2. **Uses Native FPS**: Uses the detected FPS for encoding/decoding operations
3. **Fallback Support**: Falls back to environment variable if FPS detection fails

## Implementation Details

### Modified Files

#### 1. `demo/backend/server/data/transcoder.py`

**Changes:**
- Modified `transcode()` function to use video's actual FPS instead of hardcoded environment variable
- Added FPS detection logic with fallback to environment variable
- Added logging to show which FPS is being used

**Key Changes:**
```python
# Before: Always used hardcoded FPS
fps = int(os.environ.get("VIDEO_ENCODE_FPS", "24"))

# After: Use video's actual FPS with fallback
video_fps = in_metadata.fps
if video_fps is None or video_fps <= 0:
    fps = int(os.environ.get("VIDEO_ENCODE_FPS", "24"))
    print(f"Warning: Video FPS not detected, using fallback FPS: {fps}")
else:
    fps = int(round(video_fps))
    print(f"Using video FPS: {fps} (detected from video metadata)")
```

#### 2. `docker-compose.yaml`

**Changes:**
- Added comments explaining that `VIDEO_ENCODE_FPS` now serves as a fallback
- Documented that FPS is automatically detected from uploaded videos

### How It Works

1. **Video Upload**: User uploads a video file
2. **Metadata Extraction**: `get_video_metadata()` extracts video properties including FPS
3. **FPS Detection**: The system checks if FPS is available and valid
4. **Dynamic Encoding**: `transcode()` uses the detected FPS for ffmpeg encoding
5. **Fallback**: If FPS detection fails, falls back to `VIDEO_ENCODE_FPS` environment variable

### FFmpeg Command Changes

The ffmpeg command now uses the detected FPS:

```bash
# Before (hardcoded 24 FPS)
ffmpeg ... -vf "fps=24,scale=1280:720,setsar=1:1" ...

# After (dynamic FPS from video)
ffmpeg ... -vf "fps={detected_fps},scale=1280:720,setsar=1:1" ...
```

## Benefits

### 1. **Preserved Video Quality**
- No unnecessary frame rate conversion
- Maintains original video characteristics
- Reduces encoding artifacts

### 2. **Better Performance**
- Avoids unnecessary frame interpolation/dropping
- Faster processing for videos at their native FPS
- Reduced computational overhead

### 3. **Flexibility**
- Supports videos of any FPS (30, 60, 24, etc.)
- Automatic adaptation to different video sources
- Maintains backward compatibility with fallback

### 4. **User Experience**
- Videos look more natural at their original frame rates
- Consistent behavior across different video types
- No manual configuration required

## Testing

### Manual Testing Steps

1. **Upload Different FPS Videos**:
   - Upload a 30 FPS video
   - Upload a 60 FPS video  
   - Upload a 24 FPS video

2. **Verify Processing**:
   - Check backend logs for FPS detection messages
   - Verify videos play at their original frame rates
   - Ensure no quality degradation

3. **Check Logs**:
   ```
   Using video FPS: 30 (detected from video metadata)
   Using video FPS: 60 (detected from video metadata)
   Warning: Video FPS not detected, using fallback FPS: 24
   ```

### Expected Behavior

- **30 FPS Video**: Should be processed at 30 FPS
- **60 FPS Video**: Should be processed at 60 FPS  
- **24 FPS Video**: Should be processed at 24 FPS
- **Corrupted/Unknown FPS**: Should fall back to 24 FPS with warning

## Configuration

### Environment Variables

- `VIDEO_ENCODE_FPS`: Now serves as fallback FPS (default: 24)
- Other encoding settings remain unchanged:
  - `VIDEO_ENCODE_CODEC`: libx264
  - `VIDEO_ENCODE_CRF`: 23
  - `VIDEO_ENCODE_MAX_WIDTH`: 1280
  - `VIDEO_ENCODE_MAX_HEIGHT`: 720

### Backward Compatibility

- Existing deployments continue to work
- Environment variable still provides fallback
- No breaking changes to API or UI

## Future Enhancements

1. **FPS Validation**: Add validation for unusual FPS values
2. **FPS Limits**: Add configurable min/max FPS limits
3. **User Override**: Allow users to override detected FPS
4. **Batch Processing**: Optimize for batch video processing
5. **Quality Metrics**: Add quality metrics for different FPS conversions

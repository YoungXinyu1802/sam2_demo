# Behavior Data Export Changes

## Summary

Modified the behavior data export functionality to:
1. Include LIT_LoRA status in the exported JSON file
2. Change filename format based on LIT_LoRA mode:
   - **LIT_LoRA enabled**: `<video_name>_LIT_<timestamp>.json`
   - **LIT_LoRA disabled**: `<video_name>_baseline_<timestamp>.json`

## Files Modified

### 1. `demo/frontend/src/common/utils/BehaviorTracker.ts`

#### Changes to `exportData()` method:
- Added `isLITEnabled?: boolean` parameter
- Added `litLoRAEnabled` field to the summary object in the exported JSON
- Added console log to show LIT_LoRA status during export

```typescript
exportData(isLITEnabled?: boolean): string {
  // ...
  const summary = {
    sessionId: this.sessionData.sessionId,
    videoName: this.sessionData.videoName,
    litLoRAEnabled: isLITEnabled ?? false,  // NEW FIELD
    // ... other fields
  };
  // ...
}
```

#### Changes to `downloadData()` method:
- Added `isLITEnabled?: boolean` parameter
- Modified filename generation logic to use video name and mode (LIT or baseline)
- Cleans up video name by extracting filename from path, removing `.mp4` extension and `gallery_` prefix
- Uses EST timezone with `mm-dd_hh-mm-ss` format for timestamp (uses hyphens instead of colons for filesystem compatibility)
- Format: `<video_name>_<mode>_<mm-dd_hh-mm-ss>.json`

```typescript
downloadData(filename?: string, isLITEnabled?: boolean): void {
  // ...
  if (!filename && this.sessionData) {
    let videoName = this.sessionData.videoName || 'unknown';
    // Extract filename from path if it's a full path
    if (videoName.includes('/')) {
      videoName = videoName.split('/').pop() || videoName;
    }
    // Remove .mp4 extension if present
    videoName = videoName.replace(/\.mp4$/i, '');
    // Remove gallery_ prefix if present
    videoName = videoName.replace(/^gallery_/i, '');
    const mode = isLITEnabled ? 'LIT' : 'baseline';
    
    // Get current date and time in EST
    const now = new Date();
    const estTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const month = String(estTime.getMonth() + 1).padStart(2, '0');
    const day = String(estTime.getDate()).padStart(2, '0');
    const hours = String(estTime.getHours()).padStart(2, '0');
    const minutes = String(estTime.getMinutes()).padStart(2, '0');
    const seconds = String(estTime.getSeconds()).padStart(2, '0');
    const timestamp = `${month}-${day}_${hours}-${minutes}-${seconds}`;
    
    filename = `${videoName}_${mode}_${timestamp}.json`;
  }
  // ...
}
```

### 2. `demo/frontend/src/common/components/video/editor/DemoVideoEditor.tsx`

#### Changes:
- Added import for `litLoRAModeEnabledAtom`
- Added `isLITLoRAModeEnabled` state variable using `useAtomValue(litLoRAModeEnabledAtom)`
- Updated `onVideoEndedAtFinalFrame()` to pass LIT status to `downloadData()`

```typescript
const isLITLoRAModeEnabled = useAtomValue(litLoRAModeEnabledAtom);

function onVideoEndedAtFinalFrame(_event: VideoEndedAtFinalFrameEvent) {
  console.log('[DemoVideoEditor] Video reached final frame, exporting behavior data');
  console.log('[DemoVideoEditor] LIT_LoRA mode enabled:', isLITLoRAModeEnabled);
  behaviorTracker.endSession();
  behaviorTracker.downloadData(undefined, isLITLoRAModeEnabled);
}
```

### 3. `demo/frontend/src/common/components/button/ExportBehaviorDataButton.tsx`

#### Changes:
- Added imports for `litLoRAModeEnabledAtom` and `useAtomValue`
- Added `isLITLoRAModeEnabled` state variable
- Updated `handleExport()` to pass LIT status to `downloadData()`

```typescript
const isLITLoRAModeEnabled = useAtomValue(litLoRAModeEnabledAtom);

const handleExport = useCallback(() => {
  behaviorTracker.endSession();
  behaviorTracker.downloadData(undefined, isLITLoRAModeEnabled);
}, [isLITLoRAModeEnabled]);
```

## Example Output

### Filename Examples:
Original video names are cleaned by extracting filename from path, removing `.mp4` extension and `gallery_` prefix.
Timestamp is formatted as EST date and time in `mm-dd_hh-mm-ss` format:

- Input: `/path/to/gallery_my_video.mp4` on Jan 25 at 2:30:45 PM EST → Output with LIT: `my_video_LIT_01-25_14-30-45.json`
- Input: `/path/to/gallery_my_video.mp4` on Jan 25 at 2:30:45 PM EST → Output without LIT: `my_video_baseline_01-25_14-30-45.json`
- Input: `some_video.mp4` on Mar 15 at 9:15:03 AM EST → Output with LIT: `some_video_LIT_03-15_09-15-03.json`
- Input: `another_video` on Dec 31 at 11:05:22 PM EST → Output with LIT: `another_video_LIT_12-31_23-05-22.json`

### JSON Structure (new field in summary):
```json
{
  "summary": {
    "sessionId": "...",
    "videoName": "video_name",
    "litLoRAEnabled": true,  // NEW FIELD
    "totalAnnotationTimeMs": 45000,
    "totalAnnotationTimeSeconds": 45,
    // ... other fields
  },
  "sessionData": { ... },
  "clicksPerFrame": { ... },
  "corrections": [ ... ],
  "frameCorrectionTimes": [ ... ]
}
```

## Testing

To test these changes:
1. Load a video in the demo
2. Enable/disable LIT_LoRA mode
3. Perform some annotations
4. Export behavior data (either manually or by reaching the final frame)
5. Check the downloaded filename matches the expected format
6. Open the JSON file and verify the `litLoRAEnabled` field in the summary section

## Backward Compatibility

The changes are backward compatible:
- The `isLITEnabled` parameter is optional (defaults to `false`)
- If `filename` is explicitly provided, it will be used as-is
- The `litLoRAEnabled` field is always included in the JSON (defaults to `false`)

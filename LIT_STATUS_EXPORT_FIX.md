# LIT Status Export Fix

## Problem

When clicking "Start Over" and changing the LIT status, then exporting the behavior data, the LIT status in the exported behavior data didn't reflect the current state. This happened even though the backend correctly applied the change of the LIT status.

## Root Cause

The issue was caused by **JavaScript closure capturing stale values** in React components:

1. **ExportBehaviorDataButton.tsx**: The `isLITLoRAModeEnabled` value was captured in the `useCallback` closure at component render time. Even though it was in the dependency array, there could be timing issues where the value wasn't updated when the export button was clicked.

2. **DemoVideoEditor.tsx**: The auto-export functionality (when video reaches final frame) had an even worse issue - the `isLITLoRAModeEnabled` value was captured in the `useEffect` closure, but it was **NOT** in the dependency array. This meant it captured the initial value (false) and never updated, even when the LIT status changed.

### Why This Happened After "Start Over"

When you click "Start Over":
1. The `ClearAllPointsInVideoButton` resets the LIT status to `false` via `setLitLoRAModeEnabled(false)`
2. The atom value is updated in the Jotai store
3. However, the export functions had already captured the old value in their closures
4. When you later enable LIT mode and export, the captured value is still the old one

## Solution

Changed both components to **read the atom value directly from the Jotai store at export time** instead of relying on closure-captured values:

### Before (Problematic Code)

```typescript
// ExportBehaviorDataButton.tsx
const isLITLoRAModeEnabled = useAtomValue(litLoRAModeEnabledAtom);

const handleExport = useCallback(() => {
  behaviorTracker.endSession();
  behaviorTracker.downloadData(undefined, isLITLoRAModeEnabled); // Uses captured value
}, [isLITLoRAModeEnabled]);
```

```typescript
// DemoVideoEditor.tsx
const isLITLoRAModeEnabled = useAtomValue(litLoRAModeEnabledAtom);

function onVideoEndedAtFinalFrame(_event: VideoEndedAtFinalFrameEvent) {
  behaviorTracker.endSession();
  behaviorTracker.downloadData(undefined, isLITLoRAModeEnabled); // Uses captured value
}
// Note: isLITLoRAModeEnabled was NOT in the useEffect dependency array!
```

### After (Fixed Code)

```typescript
// ExportBehaviorDataButton.tsx
const store = useStore();

const handleExport = useCallback(() => {
  // Read current value from store at export time
  const isLITLoRAModeEnabled = store.get(litLoRAModeEnabledAtom);
  
  behaviorTracker.endSession();
  behaviorTracker.downloadData(undefined, isLITLoRAModeEnabled);
}, [store]);
```

```typescript
// DemoVideoEditor.tsx
const store = useStore();

function onVideoEndedAtFinalFrame(_event: VideoEndedAtFinalFrameEvent) {
  // Read current value from store at export time
  const currentLITLoRAModeEnabled = store.get(litLoRAModeEnabledAtom);
  
  behaviorTracker.endSession();
  behaviorTracker.downloadData(undefined, currentLITLoRAModeEnabled);
}
// Added 'store' to useEffect dependency array
```

## Files Modified

1. **demo/frontend/src/common/components/button/ExportBehaviorDataButton.tsx**
   - Added `useStore` import from jotai
   - Changed to read atom value directly from store at export time
   - Updated dependency array to only include `store`

2. **demo/frontend/src/common/components/video/editor/DemoVideoEditor.tsx**
   - Added `useStore` import from jotai
   - Removed unused `isLITLoRAModeEnabled` variable declaration
   - Changed auto-export to read atom value directly from store at export time
   - Added `store` to useEffect dependency array

## How It Works Now

1. When you click "Start Over", the LIT status is reset to `false` in the Jotai store
2. If you then enable LIT mode, the value in the store is updated to `true`
3. When you export (either manually or automatically at final frame):
   - The export function calls `store.get(litLoRAModeEnabledAtom)`
   - This reads the **current** value from the store, not a captured value
   - The correct LIT status is passed to `behaviorTracker.downloadData()`
4. The exported JSON file contains the correct `litLoRAEnabled` field value

## Testing

To verify the fix:

1. Load a video
2. Click "Start Over" 
3. Change the LIT status (enable or disable)
4. Export behavior data (either manually or by reaching final frame)
5. Open the exported JSON file
6. Verify that the `summary.litLoRAEnabled` field matches the current LIT status
7. Verify that the filename includes the correct mode (`_LIT_` or `_baseline_`)

## Technical Notes

### Why `store.get()` Works

- `store.get(atom)` reads the current value from the Jotai store synchronously
- It always returns the latest value, not a captured/memoized value
- This is safe to use in callbacks and event handlers

### Alternative Solutions Considered

1. **Add `isLITLoRAModeEnabled` to dependency arrays**: This would cause the callbacks to be recreated every time the LIT status changes, which is unnecessary overhead and could cause other issues.

2. **Use `useAtomValue` with a ref**: This would work but is more complex and less idiomatic than using `store.get()`.

3. **Pass the value as a parameter**: This would require changing the event system, which is more invasive.

The `store.get()` approach is the cleanest and most direct solution.

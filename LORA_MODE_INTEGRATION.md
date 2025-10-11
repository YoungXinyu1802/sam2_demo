# LIT-LoRA Mode Button Integration

## Problem
The frontend LIT-LoRA Mode button was not properly communicating with the backend. When users clicked "Enable LIT-LoRA Mode", it only set a local JavaScript flag but never called the backend to actually enable `self.predictor.LIT_LoRA_mode = True`. This meant that when features were captured during `add_new_mask`, the condition `if self.LIT_LoRA_mode:` was False, so `temp_feat_for_lora["pix_feat_with_mem"]` was never populated, causing LoRA training to fail.

## Solution
Added proper HTTP endpoints for enabling/disabling LoRA mode and connected the frontend button to these endpoints.

## Changes Made

### Backend Changes

#### 1. `demo/backend/server/inference/data_types.py`
Added new request/response data types:
```python
@dataclass_json
@dataclass
class EnableLoRAModeRequest(BaseRequest):
    type: str
    session_id: str

@dataclass_json
@dataclass
class EnableLoRAModeResponse:
    success: bool
    message: str

@dataclass_json
@dataclass
class DisableLoRAModeRequest(BaseRequest):
    type: str
    session_id: str

@dataclass_json
@dataclass
class DisableLoRAModeResponse:
    success: bool
    message: str
```

#### 2. `demo/backend/server/inference/predictor.py`
Added HTTP endpoint methods:
```python
def enable_lora_mode_endpoint(self, request):
    """HTTP endpoint to enable LIT-LoRA mode"""
    from inference.data_types import EnableLoRAModeResponse
    try:
        self.enable_lora_mode()
        return EnableLoRAModeResponse(
            success=True,
            message="LIT-LoRA mode enabled"
        )
    except Exception as e:
        logger.error(f"Error enabling LoRA mode: {e}")
        return EnableLoRAModeResponse(
            success=False,
            message=f"Failed to enable LoRA mode: {str(e)}"
        )

def disable_lora_mode_endpoint(self, request):
    """HTTP endpoint to disable LIT-LoRA mode"""
    # Similar implementation for disabling
```

#### 3. `demo/backend/server/app.py`
Added HTTP routes:
```python
@app.route("/enable_lora_mode", methods=["POST"])
def enable_lora_mode() -> Response:
    data = request.json
    req = EnableLoRAModeRequest(
        type="enable_lora_mode",
        session_id=data["session_id"],
    )
    response = inference_api.enable_lora_mode_endpoint(request=req)
    return make_response(response.to_json(), 200)

@app.route("/disable_lora_mode", methods=["POST"])
def disable_lora_mode() -> Response:
    # Similar implementation for disabling
```

### Frontend Changes

#### 1. `demo/frontend/src/common/tracker/SAM2Model.ts`
Changed `enableLITLoRAMode()` and `disableLITLoRAMode()` from simple flag setters to async functions that call the backend:

```typescript
public async enableLITLoRAMode(): Promise<void> {
    const sessionId = this._session.id;
    if (!sessionId) {
      Logger.warn('No session ID for enabling LoRA mode');
      return;
    }

    try {
      const url = `${this._endpoint}/enable_lora_mode`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      Logger.info('Backend LoRA mode enabled:', result);
      
      this._litLoRAModeEnabled = true;
      Logger.info('LIT_LoRA mode enabled');
    } catch (error) {
      Logger.error('Failed to enable LoRA mode:', error);
      throw error;
    }
  }
```

#### 2. `demo/frontend/src/common/tracker/Tracker.ts`
Updated interface signatures:
```typescript
abstract enableLITLoRAMode(): Promise<void>;
abstract disableLITLoRAMode(): Promise<void>;
```

#### 3. `demo/frontend/src/common/components/video/Video.tsx`
Updated method signatures:
```typescript
enableLITLoRAMode(): Promise<void> {
  return bridge.enableLITLoRAMode();
}

disableLITLoRAMode(): Promise<void> {
  return bridge.disableLITLoRAMode();
}
```

#### 4. `demo/frontend/src/common/components/video/VideoWorkerBridge.ts`
Updated to return promises:
```typescript
enableLITLoRAMode(): Promise<void> {
  return this.sendRequest<EnableLITLoRAModeRequest>('enableLITLoRAMode');
}

disableLITLoRAMode(): Promise<void> {
  return this.sendRequest<DisableLITLoRAModeRequest>('disableLITLoRAMode');
}
```

#### 5. `demo/frontend/src/common/components/button/LITLoRAModeButton.tsx`
Updated button handler to async with error handling:
```typescript
const handleToggleLITLoRAMode = useCallback(async () => {
  if (isDisabled) {
    return;
  }

  if (!isLITLoRAModeEnabled) {
    // Enable LIT_LoRA mode
    enqueueMessage('frameTrackingEnabled');
    try {
      await video?.enableLITLoRAMode();
      setIsLITLoRAModeEnabled(true);
      behaviorTracker.logTrackingEvent('enable_frame_tracking');
    } catch (error) {
      console.error('Failed to enable LoRA mode:', error);
      enqueueMessage('error');
    }
  } else {
    // Disable LIT_LoRA mode
    enqueueMessage('frameTrackingDisabled');
    try {
      await video?.disableLITLoRAMode();
      setIsLITLoRAModeEnabled(false);
      behaviorTracker.logTrackingEvent('disable_frame_tracking');
    } catch (error) {
      console.error('Failed to disable LoRA mode:', error);
      enqueueMessage('error');
    }
  }
}, [isLITLoRAModeEnabled, isDisabled, video, enqueueMessage, setIsLITLoRAModeEnabled]);
```

## Workflow

Now when a user clicks the "Enable LIT-LoRA Mode" button:

1. Frontend button click → `handleToggleLITLoRAMode()` (async)
2. Calls `video.enableLITLoRAMode()` → `bridge.enableLITLoRAMode()`
3. Sends request to worker → `tracker.enableLITLoRAMode()`
4. Makes HTTP POST to `/enable_lora_mode` endpoint
5. Backend receives request → `enable_lora_mode()` is called
6. Sets `self.predictor.LIT_LoRA_mode = True`
7. Calls `self.predictor.reset_lora()` to initialize LoRA state
8. Returns success response
9. Frontend receives success → Updates UI state

## Testing

After restarting the backend and frontend:

1. Load a video and create an object mask
2. Click "Enable LIT_LoRA Mode" button
3. Check backend logs for: `"Enabling LIT-LoRA mode"`
4. Add a mask correction via `train_lora`
5. Check logs for: `"Features captured successfully. Shape: ..."`
6. The `pix_feat_with_mem` should no longer be `None`
7. LoRA training should succeed
8. Generate candidates should return results

## Key Fix

The critical fix is that **the backend's `LIT_LoRA_mode` flag is now properly set to `True`** before any corrections are made, ensuring that when `add_new_mask` calls `_track_step`, the features are captured in `temp_feat_for_lora`:

```python
# In sam2_base.py _track_step():
if self.LIT_LoRA_mode:  # Now this is True!
    self.temp_feat_for_lora["frame_idx"] = frame_idx
    self.temp_feat_for_lora["pix_feat_with_mem"] = pix_feat  # Features captured!
```

This was the root cause of the "Features not captured!" error.


# LIT_LoRA Mode Implementation

## Overview
This document describes the implementation of the LIT_LoRA (Learning In Time with Low-Rank Adaptation) mode for the SAM2 demo. This feature allows users to interactively improve mask predictions by training a LoRA adapter on corrected frames during frame-by-frame tracking.

## Features

### 1. LIT_LoRA Mode Button
- Added a new button in the toolbar to enable/disable LIT_LoRA mode
- Button is available when objects are initialized and tracking is ready
- Visual feedback with icon (Analytics when enabled, AnalyticsOff when disabled)

### 2. Interactive LoRA Training
When LIT_LoRA mode is enabled and frame-by-frame tracking is active:
- User corrections (added points) are automatically collected as training data
- Frame embeddings and masks are sent to the backend for LoRA training
- Training data accumulates across multiple frame corrections

### 3. Automatic Mask Generation
When the user pauses during frame-by-frame tracking with LIT_LoRA enabled:
- The system triggers LoRA training on collected samples
- LoRA-adapted model generates mask candidates for the current frame
- Generated masks are automatically applied to improve tracking quality

## Implementation Details

### Frontend Changes

#### 1. State Management (`demo/atoms.ts`)
```typescript
// New atoms for LIT_LoRA mode
export const litLoRAModeEnabledAtom = atom<boolean>(false);
export const loraTrainingDataAtom = atom<Array<{
  frameIndex: number;
  objectId: number;
  mask: DataArray;
}>>([]);
export const loraMaskCandidatesAtom = atom<Array<{
  objectId: number;
  mask: DataArray;
  confidence: number;
}> | null>(null);
```

#### 2. UI Component (`button/LITLoRAModeButton.tsx`)
- New button component for toggling LIT_LoRA mode
- Integrates with existing toolbar and messaging system
- Provides user feedback through snackbar messages

#### 3. Tracker Integration (`tracker/SAM2Model.ts`)
- Added `_litLoRAModeEnabled` flag
- `enableLITLoRAMode()` / `disableLITLoRAMode()` methods
- `_sendLoRATrainingData()`: Sends corrected frames to backend for training
- `_generateLoraCandidates()`: Generates and applies LoRA-predicted masks on pause

#### 4. Tracker Interface Updates
- Updated `ITracker` interface and `Tracker` abstract class
- Added LIT_LoRA methods to all tracker implementations
- Created new request types: `EnableLITLoRAModeRequest`, `DisableLITLoRAModeRequest`

#### 5. Video Worker Integration
- Added message handlers in `VideoWorker.ts` for LIT_LoRA mode actions
- Updated `VideoWorkerBridge.ts` with LIT_LoRA mode methods
- Integrated with `Video.tsx` component API

### Backend Changes

#### 1. Data Types (`inference/data_types.py`)
```python
# New request types
@dataclass
class TrainLoRARequest(BaseRequest):
    type: str
    session_id: str
    object_id: int
    frame_index: int
    mask: Mask

@dataclass
class GenerateLoraCandidatesRequest(BaseRequest):
    type: str
    session_id: str
    object_id: int
    frame_index: int

# New response types
@dataclass
class TrainLoRAResponse:
    success: bool
    message: str

@dataclass
class GenerateLoraCandidatesResponse:
    frame_index: int
    object_id: int
    candidates: List[LoRACandidateValue]
```

#### 2. Inference API (`inference/predictor.py`)
- Added LoRA training data storage: `self.lora_training_data`
- Added LoRA models storage: `self.lora_models`

**`train_lora()` method:**
- Extracts frame embeddings (image features, position encodings)
- Computes sparse and dense embeddings from masks
- Stores training samples for later LoRA training

**`generate_lora_candidates()` method:**
- Creates LoRA-adapted mask decoder (deep copy of original)
- Trains LoRA using collected samples with early stopping
- Generates mask predictions for requested frames
- Returns candidates with confidence scores

#### 3. API Endpoints (`server/app.py`)
```python
# New endpoints
@app.route("/train_lora", methods=["POST"])
def train_lora() -> Response:
    # Adds training sample for LoRA

@app.route("/generate_lora_candidates", methods=["POST"])
def generate_lora_candidates() -> Response:
    # Trains LoRA and generates mask candidates
```

#### 4. LoRA Module Integration
The implementation uses the existing `sam2_video_predictor_lora.py` module which provides:
- `LoRALinear`: Low-rank adaptation layers
- `train_with_random_split_each_epoch()`: Training loop with early stopping
- `predict()`: LoRA-adapted inference
- Loss functions: Focal loss and Dice loss

## Workflow

### User Experience Flow
1. User starts video tracking and enables frame-by-frame tracking
2. User clicks "Enable LIT_LoRA Mode" button
3. During frame-by-frame playback:
   - User notices incorrect mask on a frame
   - User pauses and adds correction points
   - System automatically sends frame and mask to backend for training
   - User continues to next frame
4. After multiple corrections, user pauses
5. System automatically:
   - Trains LoRA on collected corrections
   - Generates improved mask for current frame
   - Applies improved mask to tracking

### Technical Flow
```
User Correction (Add Points)
    ↓
Frontend: SAM2Model.updatePoints()
    ↓
Backend: InferenceAPI.add_points()
    ↓
If LIT_LoRA + Frame Tracking enabled:
    Frontend: _sendLoRATrainingData()
    ↓
    Backend: InferenceAPI.train_lora()
    ↓
    Store training sample
    
User Pauses
    ↓
Frontend: SAM2Model.logPauseEvent()
    ↓
If LIT_LoRA + Frame Tracking enabled:
    Frontend: _generateLoraCandidates()
    ↓
    Backend: InferenceAPI.generate_lora_candidates()
    ↓
    Train LoRA on collected samples
    ↓
    Generate mask predictions
    ↓
    Frontend: Apply masks to tracklets
```

## Configuration

### LoRA Training Parameters
- **Rank**: 8 (from LoRALinear)
- **Alpha**: 16.0
- **Learning Rate**: 1e-4
- **Max Epochs**: 100
- **Patience**: 10 (early stopping)
- **Train Ratio**: 0.8
- **Loss Weights**:
  - Focal Loss: 20
  - Dice Loss: 1

### Early Stopping Criteria
- No improvement for 10 epochs AND IoU > 0.6
- IoU < 0.2 after 50 epochs
- IoU = 0 after 10 epochs

## Files Modified

### Frontend
1. `demo/frontend/src/demo/atoms.ts` - State management
2. `demo/frontend/src/common/components/button/LITLoRAModeButton.tsx` - New button component
3. `demo/frontend/src/common/components/annotations/ObjectsToolbarBottomActions.tsx` - Toolbar integration
4. `demo/frontend/src/common/tracker/Tracker.ts` - Interface updates
5. `demo/frontend/src/common/tracker/SAM2Model.ts` - Core implementation
6. `demo/frontend/src/common/tracker/TrackerTypes.ts` - Type definitions
7. `demo/frontend/src/common/components/video/VideoWorkerBridge.ts` - Worker bridge
8. `demo/frontend/src/common/components/video/VideoWorker.ts` - Worker handlers
9. `demo/frontend/src/common/components/video/Video.tsx` - Component API

### Backend
1. `demo/backend/server/inference/data_types.py` - Data types
2. `demo/backend/server/inference/predictor.py` - LoRA integration
3. `demo/backend/server/app.py` - API endpoints

## Dependencies
- Existing: `sam2/sam2_video_predictor_lora.py` (LoRA training module)
- PyTorch for LoRA training
- RLE mask encoding/decoding

## Future Enhancements
1. **Multiple Candidate Display**: Show multiple LoRA-generated candidates for user selection
2. **Training Progress**: Display training progress/status to user
3. **Confidence Thresholds**: Allow user to set confidence thresholds for automatic mask application
4. **LoRA Model Persistence**: Save trained LoRA models for reuse across sessions
5. **Batch Training**: Optimize training by batching multiple corrections
6. **Real-time Training**: Train incrementally as corrections are made
7. **Model Comparison**: Show side-by-side comparison of original vs LoRA-adapted predictions

## Testing Recommendations
1. Test with various video types and object sizes
2. Verify training data collection with multiple corrections
3. Test early stopping behavior with different IoU scenarios
4. Validate mask quality improvements with LoRA
5. Test UI responsiveness during LoRA training
6. Verify memory cleanup after session closure


# LoRA Candidate Selection System

## Overview
Changed LoRA workflow to show ALL candidates to the user for selection, instead of auto-applying the best one.

## Architecture Changes

### 1. Predictor Methods (sam2_video_predictor.py)

#### `lora_predict_all_candidates(inference_state, frame_idx)` - Lines 1302-1351
**Purpose**: Generate predictions from ALL trained LoRA models

**Returns**: List of candidates with:
- `lora_idx`: Index of the LoRA model
- `low_res_masks`: Low resolution mask predictions
- `high_res_masks`: High resolution mask predictions  
- `object_score_logits`: Object confidence scores
- `predicted_mask_score`: Final mask scores at video resolution

**Key Features**:
- ✅ No GT mask required (uses dummy mask)
- ✅ Returns ALL candidates (not just best)
- ✅ Does NOT auto-apply to memory
- ✅ Read-only operation

#### `apply_lora_candidate(inference_state, frame_idx, candidate_idx)` - Lines 1353-1419
**Purpose**: Apply a selected candidate to memory (like a mask correction)

**Workflow**:
1. Re-generate prediction for the selected LoRA model
2. Encode to memory via `_encode_new_memory`
3. Update `inference_state["output_dict_per_obj"]`
4. Return applied mask for confirmation

**Key Features**:
- ✅ Acts like mask correction
- ✅ Updates memory and inference state
- ✅ Enables tracking to continue with corrected mask

### 2. Backend Endpoints (demo/backend/server/inference/predictor.py)

#### `generate_lora_candidates()` - Lines 541-634
**Purpose**: Generate ALL candidates for user selection

**Workflow**:
1. Propagate to frame to capture features
2. Call `predictor.lora_predict_all_candidates()`
3. Convert each candidate to RLE format
4. Return ALL candidates with index as identifier

**Response Format**:
```python
GenerateLoraCandidatesResponse(
    frame_index=frame_idx,
    object_id=obj_id,
    candidates=[
        LoRACandidateValue(
            mask=Mask(size=..., counts=...),
            confidence=0.0  # Index 0
        ),
        LoRACandidateValue(
            mask=Mask(size=..., counts=...),
            confidence=1.0  # Index 1
        ),
        # ... more candidates
    ]
)
```

**Note**: `confidence` field is repurposed to store the candidate index for identification.

#### `apply_lora_candidate()` - Lines 636-688
**Purpose**: Apply user-selected candidate

**Request**:
```python
ApplyLoraCandidateRequest(
    type="apply_lora_candidate",
    session_id=...,
    object_id=...,
    frame_index=...,
    candidate_index=0  # User's selection
)
```

**Response**:
```python
ApplyLoraCandidateResponse(
    success=True,
    message="Candidate 0 applied successfully"
)
```

### 3. Data Types (data_types.py)

#### New Request Type:
```python
@dataclass
class ApplyLoraCandidateRequest(BaseRequest):
    type: str
    session_id: str
    object_id: int
    frame_index: int
    candidate_index: int
```

#### New Response Type:
```python
@dataclass
class ApplyLoraCandidateResponse:
    success: bool
    message: str
```

## User Workflow

### Step 1: User Adds Correction
```
User draws correction → POST /train_lora
```
- Backend: Trains LoRA model immediately
- LoRA models stored in `predictor.multi_lora[]`

### Step 2: User Requests Candidates
```
User navigates to frame → POST /generate_lora_candidates
```
- Backend: Returns ALL LoRA predictions as candidates
- Frontend: Overlays masks with different colors/numbers

### Step 3: User Selects Candidate
```
User clicks on candidate #N → POST /apply_lora_candidate
```
- Backend: Applies candidate to memory
- Tracking continues with corrected mask

### Step 4: User Rejects All
```
User clicks "Reject All" → No action
```
- Candidates are discarded
- User can add manual correction instead

## Frontend Implementation Requirements

### Display Candidates
1. Overlay multiple masks with distinct colors
2. Show numbers/labels on each mask
3. Make masks clickable

### Color Scheme Example
```javascript
const candidateColors = [
  'rgba(255, 0, 0, 0.3)',    // Red - Candidate 0
  'rgba(0, 255, 0, 0.3)',    // Green - Candidate 1
  'rgba(0, 0, 255, 0.3)',    // Blue - Candidate 2
  'rgba(255, 255, 0, 0.3)',  // Yellow - Candidate 3
  // ... more colors
];
```

### UI Components
1. **Candidate Overlays**: 
   - Draw each mask with unique color
   - Add number label (1, 2, 3, ...)
   - Click handler for selection

2. **Action Buttons**:
   ```
   [Apply Candidate 1] [Apply Candidate 2] ... [Reject All]
   ```

3. **Visual Feedback**:
   - Highlight selected candidate
   - Show confirmation after applying
   - Update tracking visualization

### API Calls

**Request Candidates**:
```javascript
const response = await fetch('/generate_lora_candidates', {
  method: 'POST',
  body: JSON.stringify({
    type: 'generate_lora_candidates',
    session_id: sessionId,
    object_id: objectId,
    frame_index: frameIndex
  })
});
const {candidates} = await response.json();
// candidates[i].confidence contains the index
```

**Apply Selected Candidate**:
```javascript
const response = await fetch('/apply_lora_candidate', {
  method: 'POST',
  body: JSON.stringify({
    type: 'apply_lora_candidate',
    session_id: sessionId,
    object_id: objectId,
    frame_index: frameIndex,
    candidate_index: selectedIndex  // 0, 1, 2, ...
  })
});
```

## Benefits

1. **User Control**: User decides which prediction to apply
2. **Transparency**: User sees all LoRA outputs, not just best
3. **Flexibility**: Can reject all and add manual correction
4. **Comparison**: Can visually compare different LoRA predictions
5. **Safety**: No auto-apply prevents wrong corrections

## Technical Details

### Feature Capture
Before generating candidates, `propagate_to_frame()` is called to ensure features are captured in `temp_feat_for_lora`.

### Memory Management
- Candidates are temporary (not stored)
- Only selected candidate is added to memory
- Rejected candidates are discarded

### Multi-LoRA Support
- System supports multiple LoRA models (`multi_lora[]`)
- Each model produces one candidate
- Future: Could train multiple LoRAs with different data

## Testing

1. **Add correction**: Verify LoRA trains successfully
2. **Request candidates**: Should return multiple masks
3. **Visual inspection**: Masks should overlay with different colors
4. **Apply candidate**: Selected mask should be added to memory
5. **Continue tracking**: Tracking should use corrected mask
6. **Reject all**: Should allow manual correction instead

## Files Modified

1. **sam2/sam2_video_predictor.py**:
   - Added `lora_predict_all_candidates()` (lines 1302-1351)
   - Added `apply_lora_candidate()` (lines 1353-1419)
   - Removed old `lora_predict()` that auto-applied

2. **demo/backend/server/inference/predictor.py**:
   - Updated `generate_lora_candidates()` (lines 541-634)
   - Added `apply_lora_candidate()` (lines 636-688)
   - Updated imports (line 20-21)

3. **demo/backend/server/inference/data_types.py**:
   - Added `ApplyLoraCandidateRequest` (lines 150-157)
   - Added `ApplyLoraCandidateResponse` (lines 244-248)


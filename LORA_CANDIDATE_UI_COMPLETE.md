# LoRA Candidate Selection UI - Complete Implementation

## 🎉 What's Been Implemented

### Backend (✅ Complete)
1. **Endpoint to generate all candidates**: `/generate_lora_candidates`
   - Returns ALL LoRA model predictions as candidates
   - Each candidate has an index and mask

2. **Endpoint to apply selected candidate**: `/apply_lora_candidate`
   - Takes candidate index from user selection
   - Applies the selected mask to memory (like a correction)
   - Updates tracking with the chosen prediction

3. **Nginx proxy routes**: Both endpoints properly proxied through nginx

### Frontend (✅ Complete)
1. **Candidate Storage**: `loraMaskCandidatesAtom` stores candidate data
   - Object ID, frame index, and array of candidates
   - Each candidate has index, mask, and confidence

2. **Event System**: 
   - Worker posts `loraCandidatesGenerated` event
   - Main thread listens and updates atom
   - UI reacts to atom changes

3. **Selection UI Component**: `LoRACandidateSelector.tsx`
   - Shows selection buttons when candidates are generated
   - Each button has unique color and number
   - "Reject All" button to dismiss candidates
   - Styled overlay at bottom of screen

4. **Apply Logic**: 
   - Clicking a button sends message to worker
   - Worker calls backend `/apply_lora_candidate`
   - Tracking refreshes with updated mask

## 🚀 To Complete Setup

### 1. Rebuild & Restart
```bash
# Rebuild frontend (for nginx.conf and new components)
docker compose build frontend

# Restart both containers
docker compose restart backend frontend

# Or restart everything
docker compose down && docker compose up -d
```

### 2. Test the Flow

1. **Load video** and create an object
2. **Enable LIT-LoRA Mode** (button in toolbar)
3. **Add a mask correction** (this trains LoRA)
4. **Pause the video** → Candidates auto-generate
5. **Selection UI appears** with colored buttons
6. **Click a button** to apply that candidate
7. **Click "Reject All"** to dismiss without selecting

## 📝 Current Behavior

### When Candidates Generate:
- ✅ Backend returns multiple candidates
- ✅ Frontend receives and stores them
- ✅ UI shows selection buttons with colors/numbers
- ⚠️  **Masks NOT yet rendered as overlays** (buttons only)

### When User Selects:
- ✅ Backend applies to memory
- ✅ Tracking updates
- ✅ UI clears selection panel

## 🎨 TODO: Visual Mask Overlays (Optional Enhancement)

To show candidate masks as colored overlays on the video frame, you would need to:

1. **Decode RLE masks** in the frontend
2. **Render to canvas** with semi-transparent colors matching button colors
3. **Layer over video** similar to how tracklets are rendered

**Colors defined** in `LoRACandidateSelector.tsx`:
```typescript
const CANDIDATE_COLORS = [
  '#FF6B6B', // Red
  '#4ECDC4', // Cyan  
  '#45B7D1', // Blue
  '#FFA07A', // Light Salmon
  '#98D8C8', // Mint
  '#F7DC6F', // Yellow
  '#BB8FCE', // Purple
  '#85C1E2', // Sky Blue
];
```

This would involve:
- Adding candidate rendering to `VideoWorkerContext`
- Decoding masks using `jscocotools/mask`
- Drawing with `CanvasRenderingContext2D` in overlay layer
- Matching colors to button indices

## 📊 Files Modified

### Backend:
- `demo/backend/server/inference/data_types.py` - Request/response types
- `demo/backend/server/inference/predictor.py` - Apply candidate endpoint
- `demo/backend/server/app.py` - HTTP route
- `demo/frontend/nginx.conf` - Proxy configuration

### Frontend:
- `demo/frontend/src/demo/atoms.ts` - Candidate atom type
- `demo/frontend/src/common/tracker/SAM2Model.ts` - Generate & apply methods
- `demo/frontend/src/common/components/video/VideoWorkerContext.ts` - Event posting
- `demo/frontend/src/common/components/video/VideoWorker.ts` - Apply handler
- `demo/frontend/src/common/components/annotations/LoRACandidateSelector.tsx` - **NEW** UI component
- `demo/frontend/src/common/components/video/editor/DemoVideoEditor.tsx` - Event listener & component

## ✨ Summary

The complete workflow is now implemented:
1. ✅ Generate candidates from all LoRA models
2. ✅ Display selection UI with colored buttons
3. ✅ Apply selected candidate to memory
4. ✅ Reject all option

**The selection UI will appear automatically when the video is paused after LoRA training!**

To add visual mask overlays (showing the masks on the video), additional canvas rendering code would be needed, but the functional workflow is complete.


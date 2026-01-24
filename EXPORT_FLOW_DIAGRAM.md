# Behavior Data Export Flow - Final Frame Trigger

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Video Playback Begins                            │
│                  (User clicks Play button)                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                VideoWorkerContext.play()                             │
│  - Initializes requestAnimationFrame loop                           │
│  - Tracks frame progression                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              updateFrame() - Called every frame                      │
│                                                                      │
│  Calculate: calculatedFrame = Math.floor(diff/timePerFrame) + offset│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                      ▼                 ▼
         ┌────────────────────┐  ┌──────────────────┐
         │ calculatedFrame <  │  │ calculatedFrame  │
         │   numFrames        │  │  >= numFrames    │
         │                    │  │                  │
         │ Normal Playback    │  │ FINAL FRAME!     │
         └────────┬───────────┘  └────────┬─────────┘
                  │                       │
                  │                       ▼
                  │        ┌────────────────────────────────┐
                  │        │ 1. Update to final frame       │
                  │        │ 2. Process frame tracking      │
                  │        │ 3. Draw final frame           │
                  │        │ 4. Call this.pause()          │
                  │        └────────┬───────────────────────┘
                  │                 │
                  │                 ▼
                  │        ┌────────────────────────────────┐
                  │        │ sendResponse(                  │
                  │        │   'videoEndedAtFinalFrame'     │
                  │        │ )                              │
                  │        └────────┬───────────────────────┘
                  │                 │
                  │                 ▼
                  │        ┌────────────────────────────────┐
                  │        │   VideoWorkerBridge            │
                  │        │   - Receives message           │
                  │        │   - Triggers event             │
                  │        └────────┬───────────────────────┘
                  │                 │
                  │                 ▼
                  │        ┌────────────────────────────────┐
                  │        │   DemoVideoEditor              │
                  │        │   onVideoEndedAtFinalFrame()   │
                  │        └────────┬───────────────────────┘
                  │                 │
                  │                 ▼
                  │        ┌────────────────────────────────┐
                  │        │   behaviorTracker.endSession() │
                  │        │   - Records end timestamp      │
                  │        └────────┬───────────────────────┘
                  │                 │
                  │                 ▼
                  │        ┌────────────────────────────────┐
                  │        │ behaviorTracker.downloadData() │
                  │        │ - Creates JSON blob            │
                  │        │ - Triggers browser download    │
                  │        └────────────────────────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │  Continue loop until     │
    │  final frame is reached  │
    └──────────────────────────┘
```

## Key Components

### 1. VideoWorkerContext (Worker Thread)
**File:** `VideoWorkerContext.ts`
- Runs in a Web Worker for performance
- Manages video playback loop
- Detects when calculatedFrame >= numFrames
- Sends `videoEndedAtFinalFrame` message to main thread

### 2. VideoWorkerBridge (Main Thread)
**File:** `VideoWorkerBridge.ts`
- Bridge between worker and main thread
- Receives worker messages
- Emits events that components can listen to

### 3. DemoVideoEditor (React Component)
**File:** `DemoVideoEditor.tsx`
- Listens for `videoEndedAtFinalFrame` event
- Triggers behavior data export
- Manages UI state

### 4. BehaviorTracker (Data Collection)
**File:** `BehaviorTracker.ts`
- Collects user interaction data throughout session
- Stores clicks, tracking events, timestamps
- Exports data as JSON file

## Data Collected in Export

The exported JSON file includes:

```json
{
  "summary": {
    "sessionId": "...",
    "videoName": "...",
    "totalDurationMs": 45000,
    "totalDurationSeconds": 45,
    "totalClicks": 12,
    "totalCorrections": 3,
    "framesWithClicks": 5,
    "trackingEventsCount": 8
  },
  "sessionData": {
    "sessionId": "...",
    "startTime": 1706097600000,
    "endTime": 1706097645000,
    "clicks": [...],
    "trackingEvents": [...]
  },
  "clicksPerFrame": {
    "0": [...],
    "5": [...],
    ...
  },
  "corrections": [...]
}
```

## Behavior Changes

### Before Implementation
```
Frame 0 → Frame 1 → ... → Frame N-1 → Frame 0 (loops forever)
                                      ↑
                                      User must manually click
                                      "Export Behavior Data"
```

### After Implementation
```
Frame 0 → Frame 1 → ... → Frame N-1 → STOP
                                      ↓
                                      Automatic Export
                                      ↓
                                      behavior-tracking-{timestamp}.json
                                      downloaded to user's computer
```

## Thread Communication

```
┌─────────────────┐                    ┌──────────────────┐
│  Worker Thread  │                    │   Main Thread    │
│                 │                    │                  │
│ VideoWorker     │  postMessage()     │ VideoWorkerBridge│
│ Context         ├───────────────────>│                  │
│                 │  {action:          │                  │
│                 │   'videoEndedAt    │ addEventListener  │
│                 │    FinalFrame'}    │        │         │
└─────────────────┘                    └────────┼─────────┘
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │ DemoVideoEditor │
                                       │ (React)         │
                                       │                 │
                                       │ onVideoEnded... │
                                       └─────────────────┘
```

## Event Type Definitions

```typescript
// VideoWorkerBridge.ts
export interface VideoEndedAtFinalFrameEvent {}

// VideoWorkerTypes.ts  
export type VideoEndedAtFinalFrameResponse = 
  Request<'videoEndedAtFinalFrame', unknown>;

// VideoWorkerEventMap
interface VideoWorkerEventMap {
  // ... other events
  videoEndedAtFinalFrame: VideoEndedAtFinalFrameEvent;
}
```

## Testing Scenarios

1. **Normal Playback → Final Frame**
   - Play video from start
   - Video stops at final frame
   - Data automatically downloads

2. **Playback from Middle → Final Frame**
   - Start playback from frame 50 (of 100)
   - Video plays to frame 99
   - Data downloads with correct timing

3. **Frame Tracking Enabled**
   - Enable frame-by-frame tracking
   - Play video with object segmentation
   - Final frame is tracked before export
   - All tracking data included in export

4. **Manual Export Still Works**
   - User can still manually export at any time
   - Manual export button remains functional
   - Multiple exports possible in one session

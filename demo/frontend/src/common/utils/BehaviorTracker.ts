/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export type ClickEvent = {
  timestamp: number;
  frameIndex: number;
  objectId: number;
  point: [number, number];
  label: number; // 1 for positive, 0 for negative
  isCorrection: boolean; // true if added during frame tracking
  correctionTimeMs?: number; // time spent correcting this frame (populated after frame transition)
  loraTrainingTimeMs?: number; // LoRA training time for this frame (if LoRA was trained)
};

export type TrackingEvent = {
  timestamp: number;
  action: 'track_objects' | 'enable_frame_tracking' | 'disable_frame_tracking' | 'play' | 'pause';
};

export type FrameCorrectionTime = {
  frameIndex: number;
  firstClickTimestamp: number;
  frameExitTimestamp: number;
  correctionTimeMs: number;
  clickCount: number;
  loraTrainingTimeMs?: number;
};

export type SessionData = {
  sessionId: string | null;
  startTime: number;
  endTime: number | null;
  clicks: ClickEvent[];
  trackingEvents: TrackingEvent[];
  videoName: string | null;
  frameCorrectionTimes: FrameCorrectionTime[];
};

class BehaviorTracker {
  private sessionData: SessionData | null = null;
  private currentFrameFirstClickTimestamp: number | null = null;
  private currentFrameIndex: number | null = null;
  private currentFrameClickCount: number = 0;
  private currentFrameLoraTrainingTime: number | null = null;
  private loraTrainingTimesByFrame: Map<number, number> = new Map();
  private sessionCounter: number = 0; // Track how many times we've created a session

  startSession(sessionId: string | null, videoName: string | null): void {
    this.sessionCounter++;
    const sessionNumber = this.sessionCounter;
    
    console.log(`[BehaviorTracker] 🔄 Starting new session #${sessionNumber}:`, sessionId, 'for video:', videoName);
    
    // Create completely new arrays (not reusing any references)
    const newClicks: ClickEvent[] = [];
    const newTrackingEvents: TrackingEvent[] = [];
    const newFrameCorrectionTimes: FrameCorrectionTime[] = [];
    
    console.log(`[BehaviorTracker] Created new arrays - clicks ID:`, newClicks, 'frameCorrectionTimes ID:', newFrameCorrectionTimes);
    
    this.sessionData = {
      sessionId,
      startTime: 0, // Will be set when frame tracking is enabled
      endTime: null,
      clicks: newClicks,
      trackingEvents: newTrackingEvents,
      videoName,
      frameCorrectionTimes: newFrameCorrectionTimes,
    };
    
    // Store the session number for debugging
    (this.sessionData as any)._sessionNumber = sessionNumber;
    
    this.currentFrameFirstClickTimestamp = null;
    this.currentFrameIndex = null;
    this.currentFrameClickCount = 0;
    this.currentFrameLoraTrainingTime = null;
    this.loraTrainingTimesByFrame.clear();
    
    console.log(`[BehaviorTracker] ✓ New session #${sessionNumber} started with empty arrays`);
    console.log('[BehaviorTracker] clicks length:', this.sessionData.clicks.length);
    console.log('[BehaviorTracker] frameCorrectionTimes length:', this.sessionData.frameCorrectionTimes.length);
    console.log('[BehaviorTracker] sessionData reference:', this.sessionData);
  }

  logClick(
    frameIndex: number,
    objectId: number,
    point: [number, number],
    label: number,
    isCorrection: boolean = false,
  ): void {
    if (!this.sessionData) {
      console.log('[BehaviorTracker] ⚠️ logClick called but no session data!');
      return;
    }

    const timestamp = Date.now();
    
    console.log(`[BehaviorTracker] 📍 logClick: frame=${frameIndex}, obj=${objectId}, isCorrection=${isCorrection}, total clicks before: ${this.sessionData.clicks.length}`);

    this.sessionData.clicks.push({
      timestamp,
      frameIndex,
      objectId,
      point,
      label,
      isCorrection,
    });
    
    console.log(`[BehaviorTracker] 📍 Click added, total clicks now: ${this.sessionData.clicks.length}`);

    // Track correction time for this frame
    if (isCorrection) {
      if (this.currentFrameIndex !== frameIndex) {
        // New frame - reset tracking and check if we have LoRA training time for this frame
        this.currentFrameIndex = frameIndex;
        this.currentFrameFirstClickTimestamp = timestamp;
        this.currentFrameClickCount = 1;
        
        // Check if we have a stored LoRA training time for this frame
        console.log(`[BehaviorTracker] Checking for stored LoRA time for frame ${frameIndex}`);
        console.log(`[BehaviorTracker] Available frames with LoRA training:`, Array.from(this.loraTrainingTimesByFrame.keys()));
        const storedLoraTime = this.loraTrainingTimesByFrame.get(frameIndex);
        if (storedLoraTime) {
          this.currentFrameLoraTrainingTime = storedLoraTime;
          console.log(`[BehaviorTracker] ✓ Found stored LoRA training time for frame ${frameIndex}: ${storedLoraTime.toFixed(2)}ms`);
        } else {
          this.currentFrameLoraTrainingTime = null;
          console.log(`[BehaviorTracker] ✗ No stored LoRA training time for frame ${frameIndex}`);
        }
        
        console.log(`[BehaviorTracker] Started tracking correction time for frame ${frameIndex} at ${new Date(timestamp).toISOString()}`);
      } else {
        // Same frame - increment click count
        this.currentFrameClickCount++;
        console.log(`[BehaviorTracker] Additional click on frame ${frameIndex} (total: ${this.currentFrameClickCount})`);
      }
    }
  }

  logTrackingEvent(action: TrackingEvent['action']): void {
    if (!this.sessionData) {
      return;
    }

    const timestamp = Date.now();

    // Set start time when frame tracking is enabled for the first time
    if (action === 'enable_frame_tracking' && this.sessionData.startTime === 0) {
      this.sessionData.startTime = timestamp;
      console.log('[BehaviorTracker] Session start time set:', new Date(timestamp).toISOString(), '(timestamp:', timestamp + ')');
    }

    this.sessionData.trackingEvents.push({
      timestamp,
      action,
    });
  }

  logFrameTransition(newFrameIndex: number): void {
    if (!this.sessionData) {
      console.log('[BehaviorTracker] ⚠️ logFrameTransition called but no session data!');
      return;
    }

    const timestamp = Date.now();
    
    console.log(`[BehaviorTracker] 🔄 logFrameTransition: from frame ${this.currentFrameIndex} to frame ${newFrameIndex}`);

    // If we have a pending correction time to record
    if (
      this.currentFrameFirstClickTimestamp !== null &&
      this.currentFrameIndex !== null &&
      this.currentFrameIndex !== newFrameIndex
    ) {
      const correctionTimeMs = timestamp - this.currentFrameFirstClickTimestamp;
      
      console.log(`[BehaviorTracker] 📊 Recording frameCorrectionTime for frame ${this.currentFrameIndex}, total before: ${this.sessionData.frameCorrectionTimes.length}`);
      
      const frameCorrectionTime: FrameCorrectionTime = {
        frameIndex: this.currentFrameIndex,
        firstClickTimestamp: this.currentFrameFirstClickTimestamp,
        frameExitTimestamp: timestamp,
        correctionTimeMs,
        clickCount: this.currentFrameClickCount,
        loraTrainingTimeMs: this.currentFrameLoraTrainingTime || undefined,
      };

      this.sessionData.frameCorrectionTimes.push(frameCorrectionTime);
      
      console.log(`[BehaviorTracker] 📊 FrameCorrectionTime added, total now: ${this.sessionData.frameCorrectionTimes.length}`);

      // Update all correction clicks for this frame with the correction time and LoRA time
      this.sessionData.clicks.forEach(click => {
        if (click.isCorrection && click.frameIndex === this.currentFrameIndex) {
          click.correctionTimeMs = correctionTimeMs;
          if (this.currentFrameLoraTrainingTime) {
            click.loraTrainingTimeMs = this.currentFrameLoraTrainingTime;
          }
        }
      });

      const loraInfo = this.currentFrameLoraTrainingTime 
        ? `, LoRA training: ${this.currentFrameLoraTrainingTime.toFixed(0)}ms`
        : '';
      console.log(
        `[BehaviorTracker] Frame ${this.currentFrameIndex} correction completed:`,
        `${correctionTimeMs}ms with ${this.currentFrameClickCount} click(s)${loraInfo}`
      );

      // Reset for next frame
      this.currentFrameFirstClickTimestamp = null;
      this.currentFrameIndex = null;
      this.currentFrameClickCount = 0;
      this.currentFrameLoraTrainingTime = null;
    }
  }

  logLoraTrainingTime(frameIndex: number, trainingTimeMs: number): void {
    console.log(`[BehaviorTracker] logLoraTrainingTime called with frameIndex=${frameIndex}, trainingTimeMs=${trainingTimeMs}`);
    
    if (!this.sessionData) {
      console.log(`[BehaviorTracker] ERROR: No session data, cannot store LoRA training time`);
      return;
    }

    const timestamp = Date.now();

    // Store the training time by frame index for later retrieval
    this.loraTrainingTimesByFrame.set(frameIndex, trainingTimeMs);
    console.log(`[BehaviorTracker] Stored LoRA training time for frame ${frameIndex}: ${trainingTimeMs.toFixed(2)}ms`);
    console.log(`[BehaviorTracker] Map now has ${this.loraTrainingTimesByFrame.size} entries:`, Array.from(this.loraTrainingTimesByFrame.entries()));
    
    // If this is for the current frame being corrected, also set it immediately
    if (this.currentFrameIndex === frameIndex) {
      this.currentFrameLoraTrainingTime = trainingTimeMs;
      console.log(`[BehaviorTracker] Also set as current frame LoRA training time`);
    } else {
      console.log(`[BehaviorTracker] Stored for later use (current frame: ${this.currentFrameIndex}, training frame: ${frameIndex})`);
      
      // Check if we already recorded a correction time for this frame (without LoRA training time)
      // If so, update it now that training has completed
      const existingCorrectionIndex = this.sessionData.frameCorrectionTimes.findIndex(
        fc => fc.frameIndex === frameIndex
      );
      
      if (existingCorrectionIndex !== -1) {
        const existingCorrection = this.sessionData.frameCorrectionTimes[existingCorrectionIndex];
        
        // Update the frame exit timestamp to be now (when training completed)
        // and recalculate the correction time to include training duration
        const updatedCorrectionTimeMs = timestamp - existingCorrection.firstClickTimestamp;
        
        this.sessionData.frameCorrectionTimes[existingCorrectionIndex] = {
          ...existingCorrection,
          frameExitTimestamp: timestamp,
          correctionTimeMs: updatedCorrectionTimeMs,
          loraTrainingTimeMs: trainingTimeMs,
        };
        
        // Also update all correction clicks for this frame
        this.sessionData.clicks.forEach(click => {
          if (click.isCorrection && click.frameIndex === frameIndex) {
            click.correctionTimeMs = updatedCorrectionTimeMs;
            click.loraTrainingTimeMs = trainingTimeMs;
          }
        });
        
        console.log(
          `[BehaviorTracker] Updated frame ${frameIndex} correction time to ${updatedCorrectionTimeMs}ms ` +
          `(was ${existingCorrection.correctionTimeMs}ms) to include LoRA training completion`
        );
      }
    }
  }

  endSession(): void {
    console.log('[BehaviorTracker] 🏁 endSession called');
    console.log('[BehaviorTracker] sessionData exists?', this.sessionData !== null);
    
    if (!this.sessionData) {
      console.log('[BehaviorTracker] ❌ No session data to end');
      return;
    }

    const sessionNumber = (this.sessionData as any)._sessionNumber || 'unknown';
    console.log(`[BehaviorTracker] Ending session #${sessionNumber}`);
    console.log('[BehaviorTracker] sessionData reference:', this.sessionData);
    console.log('[BehaviorTracker] Before endSession - clicks:', this.sessionData.clicks.length);
    console.log('[BehaviorTracker] Before endSession - frameCorrectionTimes:', this.sessionData.frameCorrectionTimes.length);
    
    this.sessionData.endTime = Date.now();
    
    if (this.sessionData.startTime > 0) {
      const duration = this.sessionData.endTime - this.sessionData.startTime;
      console.log('[BehaviorTracker] Session ended. Duration:', Math.round(duration / 1000), 'seconds (', duration, 'ms)');
    } else {
      console.log('[BehaviorTracker] Session ended, but frame tracking was never enabled (no duration recorded)');
    }
    
    console.log('[BehaviorTracker] After endSession - clicks:', this.sessionData.clicks.length);
    console.log('[BehaviorTracker] After endSession - frameCorrectionTimes:', this.sessionData.frameCorrectionTimes.length);
  }

  exportData(isLITEnabled?: boolean): string {
    if (!this.sessionData) {
      console.log('[BehaviorTracker] ❌ exportData called but no session data available');
      return JSON.stringify({error: 'No session data available'});
    }

    const sessionNumber = (this.sessionData as any)._sessionNumber || 'unknown';
    
    // Debug: Check what's in the loraTrainingTimesByFrame map
    console.log(`[BehaviorTracker] 📤 exportData called for session #${sessionNumber}`);
    console.log('[BehaviorTracker] sessionData reference:', this.sessionData);
    console.log('[BehaviorTracker] Session ID:', this.sessionData.sessionId);
    console.log('[BehaviorTracker] Total clicks:', this.sessionData.clicks.length);
    console.log('[BehaviorTracker] Total tracking events:', this.sessionData.trackingEvents.length);
    console.log('[BehaviorTracker] frameCorrectionTimes:', this.sessionData.frameCorrectionTimes.length);
    console.log('[BehaviorTracker] loraTrainingTimesByFrame size:', this.loraTrainingTimesByFrame.size);
    console.log('[BehaviorTracker] loraTrainingTimesByFrame contents:', Array.from(this.loraTrainingTimesByFrame.entries()));
    console.log('[BehaviorTracker] LIT_LoRA enabled:', isLITEnabled);

    // Calculate duration only if startTime has been set (frame tracking was enabled)
    let totalDuration = 0;
    if (this.sessionData.startTime > 0) {
      totalDuration = this.sessionData.endTime
        ? this.sessionData.endTime - this.sessionData.startTime
        : Date.now() - this.sessionData.startTime;
    }

    const clicksPerFrame = this.sessionData.clicks.reduce((acc, click) => {
      if (!acc[click.frameIndex]) {
        acc[click.frameIndex] = [];
      }
      acc[click.frameIndex].push(click);
      return acc;
    }, {} as Record<number, ClickEvent[]>);

    const corrections = this.sessionData.clicks.filter(click => click.isCorrection);
    const correctedFrames = this.sessionData.frameCorrectionTimes;
    
    // Calculate total correction time
    const totalCorrectionTimeMs = correctedFrames.reduce(
      (sum, frame) => sum + frame.correctionTimeMs,
      0
    );

    // Calculate average correction time per frame
    const avgCorrectionTimeMs = correctedFrames.length > 0
      ? totalCorrectionTimeMs / correctedFrames.length
      : 0;

    // Calculate LoRA training statistics
    const framesWithLoraTraining = correctedFrames.filter(frame => frame.loraTrainingTimeMs);
    const totalLoraTrainingTimeMs = framesWithLoraTraining.reduce(
      (sum, frame) => sum + (frame.loraTrainingTimeMs || 0),
      0
    );
    const avgLoraTrainingTimeMs = framesWithLoraTraining.length > 0
      ? totalLoraTrainingTimeMs / framesWithLoraTraining.length
      : 0;

    const summary = {
      sessionId: this.sessionData.sessionId,
      videoName: this.sessionData.videoName,
      litLoRAEnabled: isLITEnabled ?? false,
      totalAnnotationTimeMs: totalDuration,
      totalAnnotationTimeSeconds: Math.round(totalDuration / 1000),
      totalClicks: this.sessionData.clicks.length,
      totalCorrections: corrections.length,
      framesWithClicks: Object.keys(clicksPerFrame).length,
      correctedFramesCount: correctedFrames.length,
      totalCorrectionTimeMs: Math.round(totalCorrectionTimeMs),
      totalCorrectionTimeSeconds: Math.round(totalCorrectionTimeMs / 1000),
      avgCorrectionTimeMs: Math.round(avgCorrectionTimeMs),
      avgCorrectionTimeSeconds: (avgCorrectionTimeMs / 1000).toFixed(2),
      framesWithLoraTraining: framesWithLoraTraining.length,
      totalLoraTrainingTimeMs: Math.round(totalLoraTrainingTimeMs),
      totalLoraTrainingTimeSeconds: Math.round(totalLoraTrainingTimeMs / 1000),
      avgLoraTrainingTimeMs: Math.round(avgLoraTrainingTimeMs),
      avgLoraTrainingTimeSeconds: (avgLoraTrainingTimeMs / 1000).toFixed(2),
      trackingEventsCount: this.sessionData.trackingEvents.length,
    };

    const exportData = {
      summary,
      sessionData: this.sessionData,
      clicksPerFrame,
      corrections,
      frameCorrectionTimes: correctedFrames,
    };

    // Log summary to console
    console.log('\n=== Annotation Session Summary ===');
    console.log(`Session ID: ${summary.sessionId}`);
    console.log(`Video: ${summary.videoName}`);
    console.log(`LIT_LoRA Enabled: ${summary.litLoRAEnabled}`);
    console.log(`\nTotal Annotation Time: ${summary.totalAnnotationTimeSeconds}s (${summary.totalAnnotationTimeMs}ms)`);
    console.log(`Total Correction Time: ${summary.totalCorrectionTimeSeconds}s (${summary.totalCorrectionTimeMs}ms)`);
    console.log(`Average Correction Time per Frame: ${summary.avgCorrectionTimeSeconds}s (${summary.avgCorrectionTimeMs}ms)`);
    console.log(`\nCorrected Frames: ${summary.correctedFramesCount}`);
    console.log(`Total Corrections/Clicks: ${summary.totalCorrections}`);
    console.log(`Frames with Clicks: ${summary.framesWithClicks}`);
    
    if (summary.framesWithLoraTraining > 0) {
      console.log(`\n=== LoRA Training Statistics ===`);
      console.log(`Frames with LoRA Training: ${summary.framesWithLoraTraining}`);
      console.log(`Total LoRA Training Time: ${summary.totalLoraTrainingTimeSeconds}s (${summary.totalLoraTrainingTimeMs}ms)`);
      console.log(`Average LoRA Training Time: ${summary.avgLoraTrainingTimeSeconds}s (${summary.avgLoraTrainingTimeMs}ms)`);
    }
    
    if (correctedFrames.length > 0) {
      console.log('\n=== Correction Times per Frame ===');
      correctedFrames.forEach(frame => {
        const loraInfo = frame.loraTrainingTimeMs 
          ? `, LoRA: ${(frame.loraTrainingTimeMs / 1000).toFixed(2)}s`
          : '';
        console.log(
          `Frame ${frame.frameIndex}: ${(frame.correctionTimeMs / 1000).toFixed(2)}s with ${frame.clickCount} click(s)${loraInfo}`
        );
      });
    }
    console.log('==================================\n');

    return JSON.stringify(exportData, null, 2);
  }

  downloadData(filename?: string, isLITEnabled?: boolean): void {
    console.log('[BehaviorTracker] 💾 downloadData called');
    console.log('[BehaviorTracker] sessionData exists?', this.sessionData !== null);
    if (this.sessionData) {
      console.log('[BehaviorTracker] sessionData.clicks.length:', this.sessionData.clicks.length);
      console.log('[BehaviorTracker] sessionData.frameCorrectionTimes.length:', this.sessionData.frameCorrectionTimes.length);
      console.log('[BehaviorTracker] Full sessionData:', JSON.stringify({
        sessionId: this.sessionData.sessionId,
        clicksCount: this.sessionData.clicks.length,
        frameCorrectionTimesCount: this.sessionData.frameCorrectionTimes.length,
        trackingEventsCount: this.sessionData.trackingEvents.length,
      }));
    }
    
    const data = this.exportData(isLITEnabled);
    
    console.log('[BehaviorTracker] About to create blob with data length:', data.length);
    console.log('[BehaviorTracker] First 500 chars of export:', data.substring(0, 500));
    
    const blob = new Blob([data], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    
    // Generate filename based on video name and LIT status if not provided
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
    } else if (!filename) {
      filename = `behavior-tracking-${Date.now()}.json`;
    }
    
    a.download = filename;
    console.log('[BehaviorTracker] Download filename:', filename);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  reset(): void {
    console.log('[BehaviorTracker] ⚠️ RESET called - clearing all session data');
    console.log('[BehaviorTracker] Before reset - clicks:', this.sessionData?.clicks.length || 0);
    console.log('[BehaviorTracker] Before reset - frameCorrectionTimes:', this.sessionData?.frameCorrectionTimes.length || 0);
    
    this.sessionData = null;
    this.currentFrameFirstClickTimestamp = null;
    this.currentFrameIndex = null;
    this.currentFrameClickCount = 0;
    this.currentFrameLoraTrainingTime = null;
    this.loraTrainingTimesByFrame.clear();
    
    console.log('[BehaviorTracker] ✓ Reset complete - sessionData set to null');
  }
}

export const behaviorTracker = new BehaviorTracker();


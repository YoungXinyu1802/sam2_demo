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

  startSession(sessionId: string | null, videoName: string | null): void {
    this.sessionData = {
      sessionId,
      startTime: 0, // Will be set when frame tracking is enabled
      endTime: null,
      clicks: [],
      trackingEvents: [],
      videoName,
      frameCorrectionTimes: [],
    };
    this.currentFrameFirstClickTimestamp = null;
    this.currentFrameIndex = null;
    this.currentFrameClickCount = 0;
  }

  logClick(
    frameIndex: number,
    objectId: number,
    point: [number, number],
    label: number,
    isCorrection: boolean = false,
  ): void {
    if (!this.sessionData) {
      return;
    }

    const timestamp = Date.now();

    this.sessionData.clicks.push({
      timestamp,
      frameIndex,
      objectId,
      point,
      label,
      isCorrection,
    });

    // Track correction time for this frame
    if (isCorrection) {
      if (this.currentFrameIndex !== frameIndex) {
        // New frame - reset tracking
        this.currentFrameIndex = frameIndex;
        this.currentFrameFirstClickTimestamp = timestamp;
        this.currentFrameClickCount = 1;
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
      return;
    }

    const timestamp = Date.now();

    // If we have a pending correction time to record
    if (
      this.currentFrameFirstClickTimestamp !== null &&
      this.currentFrameIndex !== null &&
      this.currentFrameIndex !== newFrameIndex
    ) {
      const correctionTimeMs = timestamp - this.currentFrameFirstClickTimestamp;
      
      const frameCorrectionTime: FrameCorrectionTime = {
        frameIndex: this.currentFrameIndex,
        firstClickTimestamp: this.currentFrameFirstClickTimestamp,
        frameExitTimestamp: timestamp,
        correctionTimeMs,
        clickCount: this.currentFrameClickCount,
      };

      this.sessionData.frameCorrectionTimes.push(frameCorrectionTime);

      // Update all correction clicks for this frame with the correction time
      this.sessionData.clicks.forEach(click => {
        if (click.isCorrection && click.frameIndex === this.currentFrameIndex) {
          click.correctionTimeMs = correctionTimeMs;
        }
      });

      console.log(
        `[BehaviorTracker] Frame ${this.currentFrameIndex} correction completed:`,
        `${correctionTimeMs}ms with ${this.currentFrameClickCount} click(s)`
      );

      // Reset for next frame
      this.currentFrameFirstClickTimestamp = null;
      this.currentFrameIndex = null;
      this.currentFrameClickCount = 0;
    }
  }

  endSession(): void {
    if (!this.sessionData) {
      return;
    }

    this.sessionData.endTime = Date.now();
    
    if (this.sessionData.startTime > 0) {
      const duration = this.sessionData.endTime - this.sessionData.startTime;
      console.log('[BehaviorTracker] Session ended. Duration:', Math.round(duration / 1000), 'seconds (', duration, 'ms)');
    } else {
      console.log('[BehaviorTracker] Session ended, but frame tracking was never enabled (no duration recorded)');
    }
  }

  exportData(): string {
    if (!this.sessionData) {
      return JSON.stringify({error: 'No session data available'});
    }

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

    const summary = {
      sessionId: this.sessionData.sessionId,
      videoName: this.sessionData.videoName,
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
    console.log(`\nTotal Annotation Time: ${summary.totalAnnotationTimeSeconds}s (${summary.totalAnnotationTimeMs}ms)`);
    console.log(`Total Correction Time: ${summary.totalCorrectionTimeSeconds}s (${summary.totalCorrectionTimeMs}ms)`);
    console.log(`Average Correction Time per Frame: ${summary.avgCorrectionTimeSeconds}s (${summary.avgCorrectionTimeMs}ms)`);
    console.log(`\nCorrected Frames: ${summary.correctedFramesCount}`);
    console.log(`Total Corrections/Clicks: ${summary.totalCorrections}`);
    console.log(`Frames with Clicks: ${summary.framesWithClicks}`);
    
    if (correctedFrames.length > 0) {
      console.log('\n=== Correction Times per Frame ===');
      correctedFrames.forEach(frame => {
        console.log(
          `Frame ${frame.frameIndex}: ${(frame.correctionTimeMs / 1000).toFixed(2)}s with ${frame.clickCount} click(s)`
        );
      });
    }
    console.log('==================================\n');

    return JSON.stringify(exportData, null, 2);
  }

  downloadData(filename?: string): void {
    const data = this.exportData();
    const blob = new Blob([data], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || `behavior-tracking-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  reset(): void {
    this.sessionData = null;
    this.currentFrameFirstClickTimestamp = null;
    this.currentFrameIndex = null;
    this.currentFrameClickCount = 0;
  }
}

export const behaviorTracker = new BehaviorTracker();


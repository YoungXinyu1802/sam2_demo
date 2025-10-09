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
};

export type TrackingEvent = {
  timestamp: number;
  action: 'track_objects' | 'enable_frame_tracking' | 'disable_frame_tracking' | 'play' | 'pause';
};

export type SessionData = {
  sessionId: string | null;
  startTime: number;
  endTime: number | null;
  clicks: ClickEvent[];
  trackingEvents: TrackingEvent[];
  videoName: string | null;
};

class BehaviorTracker {
  private sessionData: SessionData | null = null;

  startSession(sessionId: string | null, videoName: string | null): void {
    this.sessionData = {
      sessionId,
      startTime: Date.now(),
      endTime: null,
      clicks: [],
      trackingEvents: [],
      videoName,
    };
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

    this.sessionData.clicks.push({
      timestamp: Date.now(),
      frameIndex,
      objectId,
      point,
      label,
      isCorrection,
    });
  }

  logTrackingEvent(action: TrackingEvent['action']): void {
    if (!this.sessionData) {
      return;
    }

    this.sessionData.trackingEvents.push({
      timestamp: Date.now(),
      action,
    });
  }

  endSession(): void {
    if (!this.sessionData) {
      return;
    }

    this.sessionData.endTime = Date.now();
  }

  exportData(): string {
    if (!this.sessionData) {
      return JSON.stringify({error: 'No session data available'});
    }

    const totalDuration = this.sessionData.endTime
      ? this.sessionData.endTime - this.sessionData.startTime
      : Date.now() - this.sessionData.startTime;

    const clicksPerFrame = this.sessionData.clicks.reduce((acc, click) => {
      if (!acc[click.frameIndex]) {
        acc[click.frameIndex] = [];
      }
      acc[click.frameIndex].push(click);
      return acc;
    }, {} as Record<number, ClickEvent[]>);

    const corrections = this.sessionData.clicks.filter(click => click.isCorrection);

    const summary = {
      sessionId: this.sessionData.sessionId,
      videoName: this.sessionData.videoName,
      totalDurationMs: totalDuration,
      totalDurationSeconds: Math.round(totalDuration / 1000),
      totalClicks: this.sessionData.clicks.length,
      totalCorrections: corrections.length,
      framesWithClicks: Object.keys(clicksPerFrame).length,
      trackingEventsCount: this.sessionData.trackingEvents.length,
    };

    const exportData = {
      summary,
      sessionData: this.sessionData,
      clicksPerFrame,
      corrections,
    };

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
  }
}

export const behaviorTracker = new BehaviorTracker();


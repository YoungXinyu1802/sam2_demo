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
import {generateThumbnail} from '@/common/components/video/editor/VideoEditorUtils';
import VideoWorkerContext from '@/common/components/video/VideoWorkerContext';
import Logger from '@/common/logger/Logger';
import {
  SAM2ModelAddNewPointsMutation,
  SAM2ModelAddNewPointsMutation$data,
} from '@/common/tracker/__generated__/SAM2ModelAddNewPointsMutation.graphql';
import {SAM2ModelClearPointsInFrameMutation} from '@/common/tracker/__generated__/SAM2ModelClearPointsInFrameMutation.graphql';
import {SAM2ModelClearPointsInVideoMutation} from '@/common/tracker/__generated__/SAM2ModelClearPointsInVideoMutation.graphql';
import {SAM2ModelCloseSessionMutation} from '@/common/tracker/__generated__/SAM2ModelCloseSessionMutation.graphql';
import {SAM2ModelRemoveObjectMutation} from '@/common/tracker/__generated__/SAM2ModelRemoveObjectMutation.graphql';
import {SAM2ModelStartSessionMutation} from '@/common/tracker/__generated__/SAM2ModelStartSessionMutation.graphql';
import {
  BaseTracklet,
  Mask,
  SegmentationPoint,
  StreamingState,
  Tracker,
  Tracklet,
} from '@/common/tracker/Tracker';
import {TrackerOptions} from '@/common/tracker/Trackers';
import {
  ClearPointsInVideoResponse,
  SessionStartFailedResponse,
  SessionStartedResponse,
  StreamingCompletedResponse,
  StreamingStartedResponse,
  StreamingStateUpdateResponse,
  TrackletCreatedResponse,
  TrackletDeletedResponse,
  TrackletsUpdatedResponse,
} from '@/common/tracker/TrackerTypes';
import {convertMaskToRGBA} from '@/common/utils/MaskUtils';
import multipartStream from '@/common/utils/MultipartStream';
import {Stats} from '@/debug/stats/Stats';
import {INFERENCE_API_ENDPOINT} from '@/demo/DemoConfig';
import {createEnvironment} from '@/graphql/RelayEnvironment';
import {
  DataArray,
  Masks,
  RLEObject,
  decode,
  encode,
  toBbox,
} from '@/jscocotools/mask';
import {THEME_COLORS} from '@/theme/colors';
import invariant from 'invariant';
import {IEnvironment, commitMutation, graphql} from 'relay-runtime';

type Options = Pick<TrackerOptions, 'inferenceEndpoint'>;

type Session = {
  id: string | null;
  tracklets: {[id: number]: Tracklet};
};

type StreamMasksResult = {
  frameIndex: number;
  rleMaskList: Array<{
    objectId: number;
    rleMask: RLEObject;
  }>;
};

type StreamMasksAbortResult = {
  aborted: boolean;
};

export class SAM2Model extends Tracker {
  private _endpoint: string;
  private _environment: IEnvironment;

  private _session: Session = {
    id: null,
    tracklets: {},
  };
  private _streamingState: StreamingState = 'none';
  private _frameTrackingEnabled: boolean = false;
  private _streamingModeEnabled: boolean = false; // Frame-by-frame streaming mode
  private _streamingStartFrame: number = 0; // Start frame for streaming
  private _litLoRAModeEnabled: boolean = false;
  private _trackingFps: number = 5; // Default tracking FPS

  private _emptyMask: RLEObject | null = null;

  private _maskCanvas: OffscreenCanvas;
  private _maskCtx: OffscreenCanvasRenderingContext2D;

  private _stats?: Stats;

  constructor(
    context: VideoWorkerContext,
    options: Options = {
      inferenceEndpoint: INFERENCE_API_ENDPOINT,
    },
  ) {
    super(context);
    this._endpoint = options.inferenceEndpoint;
    this._environment = createEnvironment(options.inferenceEndpoint);

    this._maskCanvas = new OffscreenCanvas(0, 0);
    const maskCtx = this._maskCanvas.getContext('2d');
    invariant(maskCtx != null, 'context cannot be null');
    this._maskCtx = maskCtx;
    
    console.log(`[SAM2Model] Constructor: Initial _trackingFps = ${this._trackingFps}`);
  }

  public startSession(videoPath: string): Promise<void> {
    // Reset streaming state. Force update with the true flag to make sure the
    // UI updates its state.
    this._updateStreamingState('none', true);

    return new Promise(resolve => {
      try {
        commitMutation<SAM2ModelStartSessionMutation>(this._environment, {
          mutation: graphql`
            mutation SAM2ModelStartSessionMutation($input: StartSessionInput!) {
              startSession(input: $input) {
                sessionId
              }
            }
          `,
          variables: {
            input: {
              path: videoPath,
            },
          },
          onCompleted: response => {
            const {sessionId} = response.startSession;
            console.log(`[SAM2Model] Session started with ID: ${sessionId}`);
            this._session.id = sessionId;

            this._sendResponse<SessionStartedResponse>('sessionStarted', {
              sessionId,
            });

            // Clear any tracklets from the previous session when
            // a new session is started
            this._clearTracklets();

            // Make an empty tracklet
            this.createTracklet();
            resolve();
          },
          onError: error => {
            Logger.error(error);
            this._sendResponse<SessionStartFailedResponse>(
              'sessionStartFailed',
            );
            resolve();
          },
        });
      } catch (error) {
        Logger.error(error);
        this._sendResponse<SessionStartFailedResponse>('sessionStartFailed');
        resolve();
      }
    });
  }

  public closeSession(): Promise<void> {
    const sessionId = this._session.id;

    // Do not call cleanup before retrieving the session id because cleanup
    // will reset the session id. If the order would be changed, it would
    // never execute the closeSession mutation.
    this._cleanup();

    if (sessionId === null) {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      commitMutation<SAM2ModelCloseSessionMutation>(this._environment, {
        mutation: graphql`
          mutation SAM2ModelCloseSessionMutation($input: CloseSessionInput!) {
            closeSession(input: $input) {
              success
            }
          }
        `,
        variables: {
          input: {
            sessionId,
          },
        },
        onCompleted: response => {
          const {success} = response.closeSession;
          if (success === false) {
            reject(new Error('Failed to close session'));
            return;
          }
          resolve();
        },
        onError: error => {
          Logger.error(error);
          reject(error);
        },
      });
    });
  }

  public createTracklet(): void {
    // This will return 0 for for empty tracklets and otherwise the next
    // largest number.
    const nextId =
      Object.values(this._session.tracklets).reduce(
        (prev, curr) => Math.max(prev, curr.id),
        -1,
      ) + 1;

    const newTracklet = {
      id: nextId,
      color: THEME_COLORS[nextId % THEME_COLORS.length],
      thumbnail: null,
      points: [],
      masks: [],
      isInitialized: false,
    };

    this._session.tracklets[nextId] = newTracklet;

    // Notify the main thread
    this._updateTracklets();

    this._sendResponse<TrackletCreatedResponse>('trackletCreated', {
      tracklet: newTracklet,
    });
  }

  public deleteTracklet(trackletId: number): Promise<void> {
    const sessionId = this._session.id;
    if (sessionId === null) {
      return Promise.reject('No active session');
    }

    const tracklet = this._session.tracklets[trackletId];
    invariant(
      tracklet != null,
      'tracklet for tracklet id %s not initialized',
      trackletId,
    );

    return new Promise((resolve, reject) => {
      commitMutation<SAM2ModelRemoveObjectMutation>(this._environment, {
        mutation: graphql`
          mutation SAM2ModelRemoveObjectMutation($input: RemoveObjectInput!) {
            removeObject(input: $input) {
              frameIndex
              rleMaskList {
                objectId
                rleMask {
                  counts
                  size
                }
              }
            }
          }
        `,
        variables: {
          input: {objectId: trackletId, sessionId},
        },
        onCompleted: response => {
          const trackletUpdates = response.removeObject;
          this._sendResponse<TrackletDeletedResponse>('trackletDeleted', {
            isSuccessful: true,
          });
          for (const trackletUpdate of trackletUpdates) {
            this._updateTrackletMasks(
              trackletUpdate,
              trackletUpdate.frameIndex === this._context.frameIndex,
              false, // shouldGoToFrame
            );
          }
          this._removeTrackletMasks(tracklet);
          resolve();
        },
        onError: error => {
          this._sendResponse<TrackletDeletedResponse>('trackletDeleted', {
            isSuccessful: false,
          });
          Logger.error(error);
          reject(error);
        },
      });
    });
  }

  public updatePoints(
    frameIndex: number,
    objectId: number,
    points: SegmentationPoint[],
  ): Promise<void> {
    const sessionId = this._session.id;
    if (sessionId === null) {
      return Promise.reject('No active session');
    }

    // TODO: This is not the right place to initialize the empty mask.
    // Move this into the constructor and listen to events on the context.
    // Note, the initial context.width and context.height is 0, so it needs
    // to happen based on an event, so when the video is initialized, it needs
    // to notify the tracker to update the empty mask.
    if (this._emptyMask === null) {
      // We need to round the height/width to the nearest integer since
      // Masks.toTensor() expects an integer value for the height/width.
      const tensor = new Masks(
        Math.trunc(this._context.height),
        Math.trunc(this._context.width),
        1,
      ).toDataArray();
      this._emptyMask = encode(tensor)[0];
    }

    const tracklet = this._session.tracklets[objectId];
    invariant(
      tracklet != null,
      'tracklet for object id %s not initialized',
      objectId,
    );

    // Mark session needing propagation when point is set
    // If frame tracking or streaming mode is enabled, keep state to allow continuing playback
    if (!this._frameTrackingEnabled && !this._streamingModeEnabled) {
      this._updateStreamingState('required');
    }

    // Clear all points in frame if no points are provided.
    if (points.length === 0) {
      return this.clearPointsInFrame(frameIndex, objectId);
    }
    return new Promise((resolve, reject) => {
      const normalizedPoints = points.map(p => [
        p[0] / this._context.width,
        p[1] / this._context.height,
      ]);
      const labels = points.map(p => p[2]);
      commitMutation<SAM2ModelAddNewPointsMutation>(this._environment, {
        mutation: graphql`
          mutation SAM2ModelAddNewPointsMutation($input: AddPointsInput!) {
            addPoints(input: $input) {
              frameIndex
              rleMaskList {
                objectId
                rleMask {
                  counts
                  size
                }
              }
            }
          }
        `,
        variables: {
          input: {
            sessionId,
            frameIndex,
            objectId,
            labels: labels,
            points: normalizedPoints,
            clearOldPoints: true,
          },
        },
        onCompleted: response => {
          tracklet.points[frameIndex] = points;
          tracklet.isInitialized = true;
          this._updateTrackletMasks(response.addPoints, true);
          
          // If LIT_LoRA mode and frame tracking are enabled, send training data
          // Note: Training data is now sent manually via finishCorrection()
          // instead of automatically after each click
          
          resolve();
        },
        onError: error => {
          Logger.error(error);
          reject(error);
        },
      });
    });
  }

  public addMask(
    frameIndex: number,
    objectId: number,
    rleMask: RLEObject,
  ): Promise<void> {
    const sessionId = this._session.id;
    if (sessionId === null) {
      return Promise.reject('No active session');
    }

    const tracklet = this._session.tracklets[objectId];
    invariant(
      tracklet != null,
      'tracklet for object id %s not initialized',
      objectId,
    );

    // Mark session needing propagation when mask is set
    if (!this._frameTrackingEnabled) {
      this._updateStreamingState('required');
    }

    return new Promise((resolve, reject) => {
      const url = `${this._endpoint}/add_mask`;
      const requestBody = {
        session_id: sessionId,
        frame_index: frameIndex,
        object_id: objectId,
        mask: {
          size: rleMask.size,
          counts: rleMask.counts,
        },
      };

      const headers: {[name: string]: string} = {
        'Content-Type': 'application/json',
      };

      console.log('[SAM2Model.addMask] Sending request:', {
        url,
        frameIndex,
        objectId,
        sessionId,
        rleSize: rleMask.size,
        rleCountsLength: rleMask.counts.length,
      });

      fetch(url, {
        method: 'POST',
        body: JSON.stringify(requestBody),
        headers,
      })
        .then(async response => {
          if (!response.ok) {
            const errorText = await response.text();
            console.error('[SAM2Model.addMask] Request failed:', {
              status: response.status,
              statusText: response.statusText,
              errorText,
            });
            return Promise.reject(
              new Error(`Failed to add mask: ${errorText}`),
            );
          }
          return response.json();
        })
        .then((response: any) => {
          console.log('[SAM2Model.addMask] Response received:', {
            frame_index: response.frame_index,
            num_results: response.results?.length,
            results: response.results?.map((r: any) => ({
              object_id: r.object_id,
              mask_size: r.mask?.size,
              counts_length: r.mask?.counts?.length,
            })),
          });

          const {frame_index, results} = response;
          const rleMaskList = results.map((r: any) => ({
            objectId: r.object_id,
            rleMask: {
              size: r.mask.size,
              counts: r.mask.counts,
            },
          }));

          // Update tracklet masks similar to updatePoints
          const trackletUpdate = {
            frameIndex: frame_index,
            rleMaskList: rleMaskList.map((r: any) => ({
              objectId: r.objectId,
              rleMask: {
                size: r.rleMask.size,
                counts: r.rleMask.counts,
              },
            })),
          };

          console.log('[SAM2Model.addMask] Updating tracklet masks:', {
            frameIndex: trackletUpdate.frameIndex,
            numMasks: trackletUpdate.rleMaskList.length,
            masks: trackletUpdate.rleMaskList.map((r: any) => ({
              objectId: r.objectId,
              rleSize: r.rleMask.size,
              rleCountsLength: r.rleMask.counts.length,
            })),
          });

          // Mark tracklets as initialized (similar to updatePoints)
          for (const {objectId} of rleMaskList) {
            const tracklet = this._session.tracklets[objectId];
            if (tracklet != null) {
              tracklet.isInitialized = true;
            }
          }
          
          this._updateTrackletMasks(trackletUpdate, true);
          console.log('[SAM2Model.addMask] Tracklet masks updated successfully');
          resolve();
        })
        .catch(error => {
          console.error('[SAM2Model.addMask] Error:', error);
          Logger.error(error);
          reject(error);
        });
    });
  }

  public clearPointsInFrame(
    frameIndex: number,
    objectId: number,
  ): Promise<void> {
    const sessionId = this._session.id;
    if (sessionId === null) {
      return Promise.reject('No active session');
    }

    const tracklet = this._session.tracklets[objectId];
    invariant(
      tracklet != null,
      'tracklet for object id %s not initialized',
      objectId,
    );

    // Mark session needing propagation when point is set
    this._updateStreamingState('required');

    return new Promise((resolve, reject) => {
      commitMutation<SAM2ModelClearPointsInFrameMutation>(this._environment, {
        mutation: graphql`
          mutation SAM2ModelClearPointsInFrameMutation(
            $input: ClearPointsInFrameInput!
          ) {
            clearPointsInFrame(input: $input) {
              frameIndex
              rleMaskList {
                objectId
                rleMask {
                  counts
                  size
                }
              }
            }
          }
        `,
        variables: {
          input: {
            sessionId,
            frameIndex,
            objectId,
          },
        },
        onCompleted: response => {
          tracklet.points[frameIndex] = [];
          tracklet.isInitialized = true;
          this._updateTrackletMasks(response.clearPointsInFrame, true);
          resolve();
        },
        onError: error => {
          Logger.error(error);
          reject(error);
        },
      });
    });
  }

  public clearPointsInVideo(): Promise<void> {
    const sessionId = this._session.id;
    if (sessionId === null) {
      return Promise.reject('No active session');
    }

    // Disable frame tracking and mark session needing propagation
    this.disableFrameTracking();
    this._updateStreamingState('none');

    return new Promise(resolve => {
      commitMutation<SAM2ModelClearPointsInVideoMutation>(this._environment, {
        mutation: graphql`
          mutation SAM2ModelClearPointsInVideoMutation(
            $input: ClearPointsInVideoInput!
          ) {
            clearPointsInVideo(input: $input) {
              success
            }
          }
        `,
        variables: {
          input: {
            sessionId,
          },
        },
        onCompleted: response => {
          const {success} = response.clearPointsInVideo;
          if (!success) {
            this._sendResponse<ClearPointsInVideoResponse>(
              'clearPointsInVideo',
              {isSuccessful: false},
            );
            return;
          }

          // Reset points and masks for each tracklet
          this._clearTracklets();

          // Notify the main thread
          this._context.goToFrame(this._context.frameIndex);
          this._updateTracklets();
          this._sendResponse<ClearPointsInVideoResponse>('clearPointsInVideo', {
            isSuccessful: true,
          });
          resolve();
        },
        onError: error => {
          this._sendResponse<ClearPointsInVideoResponse>('clearPointsInVideo', {
            isSuccessful: false,
          });
          Logger.error(error);
        },
      });
    });
  }

  public async streamMasks(frameIndex: number): Promise<void> {
    const sessionId = this._session.id;
    if (sessionId === null) {
      return Promise.reject('No active session');
    }
    try {
      // Get the original video FPS and set tracking FPS to match it
      const originalVideoFps = this._context.getVideoFps();
      if (originalVideoFps !== null && originalVideoFps > 0) {
        console.log(`[SAM2Model] Original video FPS: ${originalVideoFps}`);
        this.setTrackingFps(originalVideoFps);
        this._context.setTrackingFps(originalVideoFps);
        console.log(`[SAM2Model] Set frame-by-frame tracking FPS to match original video FPS: ${originalVideoFps}`);
      } else {
        console.warn(`[SAM2Model] Could not get original video FPS, using default tracking FPS: ${this._trackingFps}`);
      }

      // 1. Clear previous masks
      this._context.clearMasks();
      this._clearTrackletMasks();

      // 2. Enable streaming mode for frame-by-frame tracking during playback
      this._streamingModeEnabled = true;
      this._streamingStartFrame = frameIndex;
      
      // Set streaming state to 'partial' BEFORE sending streamingStarted
      // This ensures the play button is enabled when streaming mode starts
      this._updateStreamingState('partial');
      
      // Send streaming started event after setting state
      this._sendResponse<StreamingStartedResponse>('streamingStarted');
      
      console.log(`[SAM2Model] streamMasks: Set streaming state to 'partial'`);

      // 3. Enable frame tracking to trigger callbacks during playback
      // The existing callback in VideoWorker will call trackFrame, which we'll enhance
      // to also handle streaming mode
      // Note: We need to enable frame tracking so the callback gets called during playback
      // The frame tracking callback is already set up in VideoWorker.ts
      // We enable both the context's frame tracking and our own flag
      this._context.enableFrameTracking(true);
      // Note: We don't set _frameTrackingEnabled here because we're using streaming mode
      // The trackFrame method will check for _streamingModeEnabled first
      
      // Important: After enabling frame tracking, we need to ensure the video can play
      // The enableFrameTracking method pauses the video, but for streaming mode we want
      // it to continue playing. However, we can't play it here - the user needs to click play.
      // The frame tracking will work once the video starts playing.
      
      // 4. Track the initial frame if video is paused at or after the start frame
      const currentFrameIndex = this._context.frameIndex;
      console.log(`[SAM2Model] Streaming mode: currentFrameIndex=${currentFrameIndex}, startFrame=${frameIndex}`);
      
      if (currentFrameIndex >= frameIndex) {
        const frameInterval = this._context.getFrameSamplingInterval();
        console.log(`[SAM2Model] Streaming mode: frameInterval=${frameInterval}`);
        // Check if current frame is a sampled frame
        if (currentFrameIndex % frameInterval === 0) {
          const reindexedFrame = Math.floor(currentFrameIndex / frameInterval);
          console.log(`[SAM2Model] Streaming mode: tracking initial frame ${currentFrameIndex} (reindexed: ${reindexedFrame})`);
          await this._trackFrameForStreaming(currentFrameIndex, reindexedFrame);
        } else {
          console.log(`[SAM2Model] Streaming mode: skipping initial frame ${currentFrameIndex} (not sampled)`);
        }
      }

      console.log(`[SAM2Model] Streaming mode enabled - will track frames frame-by-frame during playback starting from frame ${frameIndex}`);
    } catch (error) {
      Logger.error(error);
      this._streamingModeEnabled = false;
      this._context.setOnFrameCallback(null);
      this._context.enableFrameTracking(false);
      throw error;
    }
  }

  public abortStreamMasks() {
    // Disable streaming mode
    this._streamingModeEnabled = false;
    
    // Disable frame tracking if it was only enabled for streaming
    if (!this._frameTrackingEnabled) {
      this._context.enableFrameTracking(false);
    }
    
    this._updateStreamingState('none');
    this._sendResponse<StreamingCompletedResponse>('streamingCompleted');
  }

  /**
   * Helper method to track a frame for streaming mode
   */
  private async _trackFrameForStreaming(
    actualFrameIndex: number,
    reindexedFrameIndex: number,
  ): Promise<void> {
    const sessionId = this._session.id;
    if (sessionId === null || !this._streamingModeEnabled) {
      return;
    }

    // Check if we have any tracklets initialized
    const hasInitializedTracklets = Object.values(this._session.tracklets).some(
      tracklet => tracklet.isInitialized
    );
    if (!hasInitializedTracklets) {
      return; // Nothing to track yet
    }

    console.log(`[SAM2Model] Streaming: tracking frame ${actualFrameIndex} (reindexed: ${reindexedFrameIndex})`);

    try {
      const url = `${this._endpoint}/propagate_to_frame`;
      const requestBody = {
        session_id: sessionId,
        frame_index: reindexedFrameIndex,
        tracking_fps: this._trackingFps,
      };

      const headers: {[name: string]: string} = {
        'Content-Type': 'application/json',
      };

      console.log(`[SAM2Model] Streaming: sending request to ${url}`, requestBody);

      const response = await fetch(url, {
        method: 'POST',
        body: JSON.stringify(requestBody),
        headers,
      });

      if (!response.ok) {
        const errorText = await response.text();
        Logger.error(`Failed to track frame ${actualFrameIndex} for streaming: ${errorText}`);
        console.error(`[SAM2Model] Streaming: request failed for frame ${actualFrameIndex}: ${errorText}`);
        return;
      }

      const jsonResponse = await response.json();
      console.log(`[SAM2Model] Streaming: received response for frame ${actualFrameIndex}`, jsonResponse);
      const maskResults = jsonResponse.results;
      const rleMaskList = maskResults.map(
        (mask: {object_id: number; mask: RLEObject}) => {
          return {
            objectId: mask.object_id,
            rleMask: mask.mask,
          };
        },
      );

      // Store the mask at the current video frame
      const result = {
        frameIndex: actualFrameIndex,
        rleMaskList,
      };

      console.log(`[SAM2Model] Streaming: updating tracklets for frame ${actualFrameIndex} with ${rleMaskList.length} masks`);
      // Update tracklets without disrupting playback
      await this._updateTrackletMasks(result, false, false);
      console.log(`[SAM2Model] Streaming: successfully updated tracklets for frame ${actualFrameIndex}`);
      
      // Check if we've reached the end of the video
      // If so, mark streaming as complete
      // Note: This would require checking video length, which we can do via context
    } catch (error) {
      Logger.error(`Error tracking frame ${actualFrameIndex} for streaming:`, error);
    }
  }


  public async trackFrame(frameIndex: number): Promise<void> {
    const sessionId = this._session.id;
    console.log(`[SAM2Model] trackFrame called with frameIndex=${frameIndex}, sessionId=${sessionId}`);
    if (sessionId === null) {
      console.log(`[SAM2Model] No session ID, skipping trackFrame`);
      return;
    }
    
    // Check if we have any tracklets initialized
    const hasInitializedTracklets = Object.values(this._session.tracklets).some(
      tracklet => tracklet.isInitialized
    );
    if (!hasInitializedTracklets) {
      return; // Nothing to track yet
    }
    
    // If streaming mode is enabled, use streaming tracking
    if (this._streamingModeEnabled) {
      console.log(`[SAM2Model] trackFrame called in streaming mode: reindexedFrame=${frameIndex}`);
      // Calculate actual video frame index from reindexed frame
      const frameInterval = this._context.getFrameSamplingInterval();
      const actualFrameIndex = frameIndex * frameInterval;
      
      console.log(`[SAM2Model] Streaming mode: actualFrameIndex=${actualFrameIndex}, startFrame=${this._streamingStartFrame}, frameInterval=${frameInterval}`);
      
      // Only track frames from the start frame onwards
      if (actualFrameIndex >= this._streamingStartFrame) {
        console.log(`[SAM2Model] Streaming mode: calling _trackFrameForStreaming for frame ${actualFrameIndex}`);
        await this._trackFrameForStreaming(actualFrameIndex, frameIndex);
      } else {
        console.log(`[SAM2Model] Streaming mode: skipping frame ${actualFrameIndex} (before start frame ${this._streamingStartFrame})`);
      }
      return;
    }
    
    // Only track if frame tracking is enabled (not streaming mode)
    if (!this._context || !this._frameTrackingEnabled) {
      return;
    }
    
    console.log(`[SAM2Model] Starting frame propagation for frame ${frameIndex} (session: ${sessionId})`);
    
    try {
      const url = `${this._endpoint}/propagate_to_frame`;
      
      // Use stored tracking FPS
      const trackingFps = this._trackingFps;
      
      console.log(`[SAM2Model] Debug: _trackingFps=${this._trackingFps}`);
      console.log(`[SAM2Model] Sending trackFrame request: frameIndex=${frameIndex}, trackingFps=${trackingFps}`);
      
      const requestBody = {
        session_id: sessionId,
        frame_index: frameIndex,
        tracking_fps: trackingFps,
      };
      
      const headers: {[name: string]: string} = {
        'Content-Type': 'application/json',
      };
      
      const response = await fetch(url, {
        method: 'POST',
        body: JSON.stringify(requestBody),
        headers,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        Logger.error(`Failed to track frame ${frameIndex}: ${errorText}`);
        return; // Don't throw, just log and continue
      }
      
      const jsonResponse = await response.json();
      console.log(`[SAM2Model] Debug: Full response for frame ${frameIndex}:`, jsonResponse);
      const maskResults = jsonResponse.results;
      console.log(`[SAM2Model] Debug: maskResults =`, maskResults);
      const rleMaskList = maskResults.map(
        (mask: {object_id: number; mask: RLEObject}) => {
          return {
            objectId: mask.object_id,
            rleMask: mask.mask,
          };
        },
      );
      
      // Store the mask at the current video frame, not the reindexed frame
      const result = {
        frameIndex: this._context.frameIndex, // Use current video frame, not reindexed frame
        rleMaskList,
      };
      
      console.log(`[SAM2Model] Frame propagation completed for frame ${frameIndex}, received ${rleMaskList.length} masks`);
      console.log(`[SAM2Model] Debug: Processed result =`, result);
      console.log(`[SAM2Model] Debug: Current video frame = ${this._context.frameIndex}`);
      console.log(`[SAM2Model] Debug: Storing mask at current video frame ${this._context.frameIndex} instead of reindexed frame ${jsonResponse.frame_index}`);
      
      // Pass false for shouldGoToFrame to avoid disrupting the playback loop
      await this._updateTrackletMasks(result, false, false);
    } catch (error) {
      Logger.error(`Error tracking frame ${frameIndex}:`, error);
      // Don't throw, just log the error to avoid breaking video playback
    }
  }

  public async enableFrameTracking(): Promise<void> {
    this._frameTrackingEnabled = true;
    this._context.enableFrameTracking(true);
    this._updateStreamingState('full');
    
    // Reinitialize session with sampled frames for better performance
    await this._reinitializeSessionForTracking();
  }

  public disableFrameTracking(): void {
    this._frameTrackingEnabled = false;
    this._context.enableFrameTracking(false);
  }

  public setTrackingFps(fps: number): void {
    console.log(`[SAM2Model] setTrackingFps called with fps=${fps}`);
    this._trackingFps = fps;
    console.log(`[SAM2Model] Tracking FPS set to ${fps}, _trackingFps is now ${this._trackingFps}`);
  }

  private async _reinitializeSessionForTracking(): Promise<void> {
    const sessionId = this._session.id;
    if (!sessionId) {
      console.log(`[SAM2Model] No session ID, skipping reinitialization`);
      return;
    }

    try {
      console.log(`[SAM2Model] Reinitializing session for tracking FPS ${this._trackingFps}`);
      
      const url = `${this._endpoint}/reinitialize_for_tracking`;
      const requestBody = {
        session_id: sessionId,
        tracking_fps: this._trackingFps,
      };
      
      const headers: {[name: string]: string} = {
        'Content-Type': 'application/json',
      };
      
      const response = await fetch(url, {
        method: 'POST',
        body: JSON.stringify(requestBody),
        headers,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[SAM2Model] Failed to reinitialize session: ${errorText}`);
        return;
      }
      
      const jsonResponse = await response.json();
      console.log(`[SAM2Model] Session reinitialized: ${jsonResponse.message}`);
      
    } catch (error) {
      console.error(`[SAM2Model] Error reinitializing session:`, error);
    }
  }

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

  public async disableLITLoRAMode(): Promise<void> {
    const sessionId = this._session.id;
    if (!sessionId) {
      Logger.warn('No session ID for disabling LoRA mode');
      return;
    }

    try {
      const url = `${this._endpoint}/disable_lora_mode`;
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
      Logger.info('Backend LoRA mode disabled:', result);
      
      this._litLoRAModeEnabled = false;
      Logger.info('LIT_LoRA mode disabled');
    } catch (error) {
      Logger.error('Failed to disable LoRA mode:', error);
      throw error;
    }
  }

  public finishCorrection(): void {
    if (!this._litLoRAModeEnabled || !this._frameTrackingEnabled) {
      Logger.warn('Cannot finish correction: LIT_LoRA mode or frame tracking not enabled');
      return;
    }

    // Send the current frame's masks for all tracklets as training data
    const frameIndex = this._context.frameIndex;
    const trackletIds = Object.keys(this._session.tracklets).map(Number);

    for (const objectId of trackletIds) {
      const tracklet = this._session.tracklets[objectId];
      if (tracklet && tracklet.isInitialized) {
        // Get the latest mask for this object at this frame
        const mask = tracklet.masks[frameIndex];
        if (mask && mask.data) {
          // mask.data can be Blob or RLEObject, we need the RLEObject
          const rleData = mask.data;
          // Type guard: check if it's an RLEObject (has size and counts properties)
          if ('size' in rleData && 'counts' in rleData) {
            const rleObject: RLEObject = {
              size: rleData.size as [number, number],
              counts: rleData.counts,
            };
            this._sendLoRATrainingData(frameIndex, objectId, rleObject);
            Logger.info(`Sent training data for object ${objectId} at frame ${frameIndex}`);
          }
        }
      }
    }
  }

  public enableStats(): void {
    this._stats = new Stats('ms', 'D', 1000 / 25);
  }

  public logPlayEvent(): void {
    // No-op: tracking handled in main thread
  }

  public logPauseEvent(): void {
    // No-op: LoRA candidate generation is now manual via button
  }

  public async generateLoraCandidates(): Promise<void> {
    const sessionId = this._session.id;
    if (!sessionId) {
      Logger.warn('No session ID for LoRA candidate generation');
      return;
    }

    try {
      // Get current frame and active tracklets
      const frameIndex = this._context.frameIndex;
      const trackletIds = Object.keys(this._session.tracklets).map(Number);

      // Generate candidates for each tracklet
      for (const objectId of trackletIds) {
        const url = `${this._endpoint}/generate_lora_candidates`;
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            session_id: sessionId,
            object_id: objectId,
            frame_index: frameIndex,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        Logger.info('LoRA candidates generated:', result);

        // Store candidates for user selection instead of auto-applying
        if (result.candidates && result.candidates.length > 0) {
          const candidates = result.candidates.map((candidate: any, index: number) => ({
            index,
            mask: candidate.mask,
            confidence: candidate.confidence,
          }));
          
          // Emit event with candidates for UI to display
          this._context.sendLoraCandidates({
            objectId,
            frameIndex,
            candidates,
          });
          
          Logger.info(`Generated ${candidates.length} LoRA candidates for user selection`);
        } else {
          Logger.warn('No LoRA candidates generated');
        }
      }
    } catch (error) {
      Logger.error('Failed to generate LoRA candidates:', error);
    }
  }

  public async applyLoraCandidate(
    objectId: number,
    frameIndex: number,
    candidateIndex: number,
  ): Promise<void> {
    const sessionId = this._session.id;
    if (!sessionId) {
      Logger.warn('No session ID for applying LoRA candidate');
      return;
    }

    try {
      const url = `${this._endpoint}/apply_lora_candidate`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          object_id: objectId,
          frame_index: frameIndex,
          candidate_index: candidateIndex,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      Logger.info('LoRA candidate applied:', result);

      // After applying, refresh the tracking to get the updated mask
      await this.trackFrame(frameIndex);
    } catch (error) {
      Logger.error('Failed to apply LoRA candidate:', error);
      throw error;
    }
  }

  public async startOver(): Promise<void> {
    const sessionId = this._session.id;
    if (!sessionId) {
      Logger.warn('No session ID for start over');
      return;
    }

    try {
      const url = `${this._endpoint}/start_over`;
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
      Logger.info('Start over completed:', result);

      // Reset frontend states
      this._litLoRAModeEnabled = false;
      this._frameTrackingEnabled = false;
      
      // Disable frame tracking in context
      this._context.enableFrameTracking(false);
      
      // Clear LoRA candidates
      this._context.clearLoraCandidates();
      
      Logger.info('All states reset to original condition');
    } catch (error) {
      Logger.error('Failed to start over:', error);
      throw error;
    }
  }

  // PRIVATE

  private async _sendLoRATrainingData(
    frameIndex: number,
    objectId: number,
    mask: RLEObject,
  ): Promise<void> {
    const sessionId = this._session.id;
    if (!sessionId) {
      Logger.warn('No session ID for LoRA training data');
      return;
    }

    try {
      const url = `${this._endpoint}/train_lora`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          object_id: objectId,
          frame_index: frameIndex,
          mask: {
            counts: mask.counts,
            size: mask.size,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      Logger.info('LoRA training data sent:', result);
    } catch (error) {
      Logger.error('Failed to send LoRA training data:', error);
    }
  }

  private _cleanup() {
    this._session.id = null;
    // Clear existing tracklets
    this._session.tracklets = [];
  }

  private _clearTracklets() {
    this._session.tracklets = [];
    this._context.clearMasks();
  }

  private _updateStreamingState(
    state: StreamingState,
    forceUpdate: boolean = false,
  ) {
    if (!forceUpdate && this._streamingState === state) {
      return;
    }
    this._streamingState = state;
    this._sendResponse<StreamingStateUpdateResponse>('streamingStateUpdate', {
      state,
    });
  }

  private async _removeTrackletMasks(tracklet: Tracklet) {
    this._context.clearTrackletMasks(tracklet);
    delete this._session.tracklets[tracklet.id];

    // Notify the main thread
    this._context.goToFrame(this._context.frameIndex);
    this._updateTracklets();
  }

  private async _updateTrackletMasks(
    data: SAM2ModelAddNewPointsMutation$data['addPoints'],
    updateThumbnails: boolean,
    shouldGoToFrame: boolean = true,
  ) {
    const {frameIndex, rleMaskList} = data;

    // 1. parse and decode masks for all objects
    console.log(`[SAM2Model] Debug: Processing ${rleMaskList.length} masks for frame ${frameIndex}`);
    console.log(`[SAM2Model] Debug: Available tracklets:`, Object.keys(this._session.tracklets));
    for (const {objectId, rleMask} of rleMaskList) {
      console.log(`[SAM2Model] Debug: Processing mask for objectId ${objectId}`);
      const track = this._session.tracklets[objectId];
      console.log(`[SAM2Model] Debug: Found tracklet for objectId ${objectId}:`, track);
      const {size, counts} = rleMask;
      const rleObject: RLEObject = {
        size: [size[0], size[1]],
        counts: counts,
      };
      const isEmpty = counts === this._emptyMask?.counts;

      this._stats?.begin();

      const decodedMask = decode([rleObject]);
      const bbox = toBbox([rleObject]);

      const mask: Mask = {
        data: rleObject as RLEObject,
        shape: [...decodedMask.shape],
        bounds: [
          [bbox[0], bbox[1]],
          [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        ],
        isEmpty,
      } as const;
      track.masks[frameIndex] = mask;
      console.log(`[SAM2Model] Debug: Stored mask at frame ${frameIndex} for tracklet ${objectId}`);

      if (updateThumbnails && !isEmpty) {
        const {ctx} = await this._compressMaskForCanvas(decodedMask);
        const frame = this._context.currentFrame as VideoFrame;
        await generateThumbnail(track, frameIndex, mask, frame, ctx);
      }
    }

    console.log(`[SAM2Model] Debug: Calling updateTracklets with frameIndex=${frameIndex}, shouldGoToFrame=${shouldGoToFrame}`);
    this._context.updateTracklets(
      frameIndex,
      Object.values(this._session.tracklets),
      shouldGoToFrame,
    );

    // Notify the main thread
    console.log(`[SAM2Model] Debug: Calling _updateTracklets to notify main thread`);
    this._updateTracklets();
  }

  private _updateTracklets() {
    const tracklets: BaseTracklet[] = Object.values(
      this._session.tracklets,
    ).map(tracklet => {
      // Notify the main thread
      const {
        id,
        color,
        isInitialized,
        points: trackletPoints,
        thumbnail,
        masks,
      } = tracklet;
      return {
        id,
        color,
        isInitialized,
        points: trackletPoints,
        thumbnail,
        masks: masks.map(mask => ({
          shape: mask.shape,
          bounds: mask.bounds,
          isEmpty: mask.isEmpty,
        })),
      };
    });

    this._sendResponse<TrackletsUpdatedResponse>('trackletsUpdated', {
      tracklets,
    });
  }

  private _clearTrackletMasks() {
    const keys = Object.keys(this._session.tracklets);
    for (const key of keys) {
      const trackletId = Number(key);
      const tracklet = {...this._session.tracklets[trackletId], masks: []};
      this._session.tracklets[trackletId] = tracklet;
    }
    this._updateTracklets();
  }

  private async _compressMaskForCanvas(
    decodedMask: DataArray,
  ): Promise<{compressedData: Blob; ctx: OffscreenCanvasRenderingContext2D}> {
    const data = convertMaskToRGBA(decodedMask.data as Uint8Array);

    this._maskCanvas.width = decodedMask.shape[0];
    this._maskCanvas.height = decodedMask.shape[1];

    const imageData = new ImageData(
      data,
      decodedMask.shape[0],
      decodedMask.shape[1],
    );
    this._maskCtx.putImageData(imageData, 0, 0);

    const canvas = new OffscreenCanvas(
      decodedMask.shape[1],
      decodedMask.shape[0],
    );

    const ctx = canvas.getContext('2d');
    invariant(ctx != null, 'context cannot be null');
    ctx.save();
    ctx.rotate(Math.PI / 2);
    // Since the image was previously rotated 90° clockwise, after the image is rotated,
    // we scale the canvas's width using scaleY and height using scaleX.
    ctx.scale(1, -1);
    ctx.drawImage(this._maskCanvas, 0, 0);
    ctx.restore();

    const compressedData = await canvas.convertToBlob({type: 'image/png'});

    return {compressedData, ctx};
  }

  // Legacy method - no longer used since we switched to frame-by-frame tracking
  // Kept for reference but not called
  // @ts-ignore - intentionally unused legacy method
  private async *_streamMasksForSession(
    abortController: AbortController,
    sessionId: string,
    startFrameIndex: undefined | number = 0,
  ): AsyncGenerator<StreamMasksResult | StreamMasksAbortResult, undefined> {
    const url = `${this._endpoint}/propagate_in_video`;

    const requestBody = {
      session_id: sessionId,
      start_frame_index: startFrameIndex,
    };

    const headers: {[name: string]: string} = Object.assign({
      'Content-Type': 'application/json',
    });

    const response = await fetch(url, {
      method: 'POST',
      body: JSON.stringify(requestBody),
      headers,
    });

    const contentType = response.headers.get('Content-Type');
    if (
      contentType == null ||
      !contentType.startsWith('multipart/x-savi-stream;')
    ) {
      throw new Error(
        'endpoint needs to support Content-Type "multipart/x-savi-stream"',
      );
    }

    const responseBody = response.body;
    if (responseBody == null) {
      throw new Error('response body is null');
    }

    const reader = multipartStream(contentType, responseBody).getReader();

    const textDecoder = new TextDecoder();

    while (true) {
      if (abortController.signal.aborted) {
        reader.releaseLock();
        yield {aborted: true};
        return;
      }

      const {done, value} = await reader.read();
      if (done) {
        return;
      }

      const {headers, body} = value;

      const contentType = headers.get('Content-Type') as string;

      if (contentType.startsWith('application/json')) {
        const jsonResponse = JSON.parse(textDecoder.decode(body));
        const maskResults = jsonResponse.results;
        const rleMaskList = maskResults.map(
          (mask: {object_id: number; mask: RLEObject}) => {
            return {
              objectId: mask.object_id,
              rleMask: mask.mask,
            };
          },
        );
        yield {
          frameIndex: jsonResponse.frame_index,
          rleMaskList,
        };
      }
    }
  }
}

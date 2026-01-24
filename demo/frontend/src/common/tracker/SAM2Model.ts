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
  MemoryInitializationStatusResponse,
  SessionStartFailedResponse,
  SessionStartedResponse,
  StreamingCompletedResponse,
  StreamingStartedResponse,
  StreamingStateUpdateResponse,
  TrackletCreatedResponse,
  TrackletDeletedResponse,
  TrackletsUpdatedResponse,
  TrainingProgressResponse,
} from '@/common/tracker/TrackerTypes';
import {convertMaskToRGBA} from '@/common/utils/MaskUtils';
import multipartStream from '@/common/utils/MultipartStream';
import {behaviorTracker} from '@/common/utils/BehaviorTracker';
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
  videoPath: string | null; // Store video path for behavior tracker restart
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

  private abortController: AbortController | null = null;
  private _session: Session = {
    id: null,
    tracklets: {},
    videoPath: null,
  };
  private _streamingState: StreamingState = 'none';
  private _frameTrackingEnabled: boolean = false;
  private _litLoRAModeEnabled: boolean = false;
  private _trackingFps: number = 5; // Default tracking FPS
  private _trainedFrames: Set<string> = new Set(); // Track which frames have been trained (format: "objectId:frameIndex")
  private _memoryEncoderInitialized: boolean = false; // Track if memory encoder has been initialized

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
            this._session.videoPath = videoPath; // Store video path for later use

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
    // If frame tracking is enabled, keep state as 'full' to allow continuing playback
    if (!this._frameTrackingEnabled) {
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
          // Pass shouldGoToFrame=false to prevent triggering train_lora on addPoints
          // train_lora should only be triggered during frame propagation
          this._updateTrackletMasks(response.addPoints, true, false);
          
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
          
          // Pass shouldGoToFrame=false to prevent triggering train_lora on addMask
          this._updateTrackletMasks(trackletUpdate, true, false);
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
          // Pass shouldGoToFrame=false to prevent triggering train_lora on clearPointsInFrame
          this._updateTrackletMasks(response.clearPointsInFrame, true, false);
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

          // Recreate default tracklet (like in startSession)
          this.createTracklet();

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
      this._sendResponse<StreamingStartedResponse>('streamingStarted');

      // 1. Clear previous masks
      this._context.clearMasks();
      this._clearTrackletMasks();

      // 2. Create abort controller and async generator
      const controller = new AbortController();
      this.abortController = controller;

      this._updateStreamingState('requesting');
      const generator = this._streamMasksForSession(
        controller,
        sessionId,
        frameIndex,
      );

      // 3. parse stream response and update masks in session objects
      let isAborted = false;
      for await (const result of generator) {
        if ('aborted' in result) {
          this._updateStreamingState('aborting');
          isAborted = true;
        } else {
          await this._updateTrackletMasks(result, false);
          this._updateStreamingState('partial');
        }
      }

      if (!isAborted) {
        // Mark session needing propagation when point is set
        this._updateStreamingState('full');
      }
    } catch (error) {
      Logger.error(error);
      throw error;
    }

    this._sendResponse<StreamingCompletedResponse>('streamingCompleted');
  }

  public abortStreamMasks() {
    this.abortController?.abort();
    this._sendResponse<StreamingCompletedResponse>('streamingCompleted');
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
    
    // Only track if frame tracking is enabled
    if (!this._context || !this._frameTrackingEnabled) {
      return;
    }
    
    console.log(`[SAM2Model] Starting frame propagation for frame ${frameIndex} (session: ${sessionId})`);
    
    try {
      // If LoRA mode is enabled, train LoRA first before propagating
      if (this._litLoRAModeEnabled) {
        console.log(`[SAM2Model] LoRA mode enabled, training LoRA before propagation`);
        await this._trainLoraBeforePropagation();
      }
      
      // Show initialization popup only on the first trackFrame call (when memory encoder needs initialization)
      const needsInitialization = !this._memoryEncoderInitialized;
      if (needsInitialization) {
        console.log(`[SAM2Model] First frame tracking call - showing initialization popup`);
        this._sendResponse<MemoryInitializationStatusResponse>('memoryInitializationStatus', {
          isInitializing: true,
        });
      }
      
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
      
      // Check if memory encoder initialization happened
      const memoryEncoderInitialized = jsonResponse.memory_encoder_initialized || false;
      console.log(`[SAM2Model] Memory encoder initialized: ${memoryEncoderInitialized}`);
      
      // If initialization happened, mark it as done and hide the popup
      if (needsInitialization) {
        this._memoryEncoderInitialized = true;
        this._sendResponse<MemoryInitializationStatusResponse>('memoryInitializationStatus', {
          isInitializing: false,
        });
        console.log(`[SAM2Model] Memory encoder initialization completed, hiding popup`);
      }
      
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
      
      // Check if LoRA candidates were auto-generated during propagation
      if (jsonResponse.lora_candidates && jsonResponse.lora_candidates.length > 0) {
        Logger.info(`[SAM2Model] Auto-displaying ${jsonResponse.lora_candidates.length} LoRA candidates from propagation`);
        
        // Get the first object ID (assuming single object for now)
        const objectId = rleMaskList.length > 0 ? rleMaskList[0].objectId : 0;
        
        const candidates = jsonResponse.lora_candidates.map((candidate: any, index: number) => ({
          index,
          mask: candidate.mask,
          confidence: candidate.confidence,
        }));
        
        // Send candidates to UI for display
        this._context.sendLoraCandidates({
          objectId,
          frameIndex: this._context.frameIndex,
          candidates,
        });
      }
    } catch (error) {
      Logger.error(`Error tracking frame ${frameIndex}:`, error);
      // Don't throw, just log the error to avoid breaking video playback
    }
  }

  public async enableFrameTracking(): Promise<void> {
    this._frameTrackingEnabled = true;
    this._context.enableFrameTracking(true);
    this._updateStreamingState('full');
    
    // Reset memory encoder initialization flag so popup shows on first frame
    this._memoryEncoderInitialized = false;
    
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
    const trainingRequests: Promise<void>[] = [];

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
            trainingRequests.push(
              this._sendLoRATrainingData(frameIndex, objectId, rleObject).then(() => {}),
            );
            Logger.info(`Sent training data for object ${objectId} at frame ${frameIndex}`);
          }
        }
      }
    }

    if (trainingRequests.length > 0) {
      void Promise.all(trainingRequests).then(() => {
        void this.generateLoraCandidates();
      });
    }
  }

  public enableStats(): void {
    this._stats = new Stats('ms', 'D', 1000 / 25);
  }

  public logPlayEvent(): void {
    // No-op: tracking handled in main thread
  }

  public logPauseEvent(): void {
    // No-op
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
      this._trainedFrames.clear(); // Clear trained frames tracking
      this._memoryEncoderInitialized = false; // Reset memory encoder initialization flag
      
      // Disable frame tracking in context
      this._context.enableFrameTracking(false);
      
      // Clear LoRA candidates
      this._context.clearLoraCandidates();
      
      // Reset behavior tracker and restart with the same session
      // NOTE: This resets the worker's instance, but the main thread also needs to be reset
      // The main thread reset will be triggered via a response event
      Logger.info('[StartOver] About to reset BehaviorTracker in worker');
      Logger.info('[StartOver] Current sessionId:', sessionId);
      Logger.info('[StartOver] Current videoPath:', this._session.videoPath);
      
      behaviorTracker.reset();
      
      if (sessionId && this._session.videoPath) {
        Logger.info('[StartOver] Restarting BehaviorTracker in worker with sessionId:', sessionId, 'videoPath:', this._session.videoPath);
        behaviorTracker.startSession(sessionId, this._session.videoPath);
        Logger.info('[StartOver] BehaviorTracker session restarted successfully in worker');
      } else {
        Logger.warn('[StartOver] Cannot restart BehaviorTracker in worker - missing sessionId or videoPath');
        Logger.warn('[StartOver] sessionId:', sessionId, 'videoPath:', this._session.videoPath);
      }
      
      // Send a response to the main thread so it can also reset its behaviorTracker
      this._sendResponse<any>('startOverCompleted', {
        sessionId: sessionId,
        videoPath: this._session.videoPath,
      });
      
      Logger.info('All states reset to original condition');
    } catch (error) {
      Logger.error('Failed to start over:', error);
      throw error;
    }
  }

  // PRIVATE
  private async _trainLoraBeforePropagation(): Promise<void> {
    const trackletIds = Object.keys(this._session.tracklets).map(Number);
    
    // Collect frames that need training (have points but haven't been trained yet)
    const framesToTrain: Array<{frameIndex: number; objectId: number; mask: RLEObject}> = [];
    
    for (const objectId of trackletIds) {
      const tracklet = this._session.tracklets[objectId];
      if (tracklet && tracklet.isInitialized) {
        // Find all frames that have user input (points) for this object
        for (let frameIndex = 0; frameIndex < tracklet.points.length; frameIndex++) {
          const points = tracklet.points[frameIndex];
          const mask = tracklet.masks[frameIndex];
          const frameKey = `${objectId}:${frameIndex}`;
          
          // Only train on frames where:
          // 1. User has added points (corrections)
          // 2. Haven't been trained yet
          if (points && points.length > 0 && mask && mask.data && !mask.isEmpty && !this._trainedFrames.has(frameKey)) {
            const rleData = mask.data;
            // Type guard: check if it's an RLEObject (has size and counts properties)
            if ('size' in rleData && 'counts' in rleData) {
              const rleObject: RLEObject = {
                size: rleData.size as [number, number],
                counts: rleData.counts,
              };
              framesToTrain.push({frameIndex, objectId, mask: rleObject});
              // Mark as trained
              this._trainedFrames.add(frameKey);
            }
          }
        }
      }
    }
    
    // Only show training message and train if there are new frames
    if (framesToTrain.length > 0) {
      this._sendResponse<TrainingProgressResponse>('trainingProgress', {
        message: 'Training LoRA model...',
      });
      
      let totalTrainingTimeMs = 0;
      for (const {frameIndex, objectId, mask} of framesToTrain) {
        const trainingTimeMs = await this._sendLoRATrainingData(frameIndex, objectId, mask);
        Logger.info(`[SAM2Model] Received training time: ${trainingTimeMs}ms for frame ${frameIndex}`);
        if (trainingTimeMs) {
          totalTrainingTimeMs += trainingTimeMs;
          // Send training time event to main thread for logging
          const eventData = {
            message: `LoRA trained in ${trainingTimeMs.toFixed(0)}ms`,
            trainingTimeMs: trainingTimeMs,
            frameIndex: frameIndex,
          };
          Logger.info(`[SAM2Model] Sending trainingProgress event with data:`, eventData);
          this._sendResponse<TrainingProgressResponse>('trainingProgress', eventData);
        } else {
          Logger.warn(`[SAM2Model] No training time received for frame ${frameIndex}`);
        }
        Logger.info(`Trained LoRA for object ${objectId} at frame ${frameIndex} before propagation`);
      }
      if (totalTrainingTimeMs > 0) {
        Logger.info(`Total LoRA training time: ${totalTrainingTimeMs.toFixed(2)}ms`);
      }
    }
  }

  private async _sendLoRATrainingData(
    frameIndex: number,
    objectId: number,
    mask: RLEObject,
  ): Promise<number | null> {
    const sessionId = this._session.id;
    if (!sessionId) {
      Logger.warn('No session ID for LoRA training data');
      return null;
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
      Logger.info('[LoRA Timing] Raw response from backend:', result);
      const trainingTimeMs = result.training_time_ms;
      Logger.info(`[LoRA Timing] Extracted training_time_ms: ${trainingTimeMs}`);
      Logger.info('LoRA training data sent:', result);
      if (trainingTimeMs) {
        Logger.info(`LoRA training completed in ${trainingTimeMs.toFixed(2)}ms`);
      } else {
        Logger.warn('[LoRA Timing] No training time received from backend!');
      }
      return trainingTimeMs || null;
    } catch (error) {
      Logger.error('Failed to send LoRA training data:', error);
      return null;
    }
  }

  private _cleanup() {
    this._session.id = null;
    this._session.videoPath = null;
    // Clear existing tracklets
    this._session.tracklets = [];
    // Clear trained frames tracking
    this._trainedFrames.clear();
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
      if (updateThumbnails && !isEmpty) {
        const {ctx} = await this._compressMaskForCanvas(decodedMask);
        const frame = this._context.currentFrame as VideoFrame;
        await generateThumbnail(track, frameIndex, mask, frame, ctx);
      }
    }

    this._context.updateTracklets(
      frameIndex,
      Object.values(this._session.tracklets),
      shouldGoToFrame,
    );

    // Notify the main thread
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

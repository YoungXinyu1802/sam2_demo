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
import useRestartSession from '@/common/components/session/useRestartSession';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import useVideo from '@/common/components/video/editor/useVideo';
import useInputVideo from '@/common/components/video/useInputVideo';
import {behaviorTracker} from '@/common/utils/BehaviorTracker';
import {
  isPlayingAtom,
  isStreamingAtom,
  labelTypeAtom,
  litLoRAModeEnabledAtom,
  frameTrackingEnabledAtom,
  loraMaskCandidatesAtom,
  loraTrainingDataAtom,
  memoryInitializedAtom,
  sessionAtom,
  correctedFramesAtom,
  sessionResetKeyAtom,
} from '@/demo/atoms';
import {Reset} from '@carbon/icons-react';
import stylex from '@stylexjs/stylex';
import {useAtomValue, useSetAtom} from 'jotai';
import {useState} from 'react';
import {Button, Loading} from 'react-daisyui';

const styles = stylex.create({
  container: {
    display: 'flex',
    alignItems: 'center',
  },
});

type Props = {
  onRestart: () => void;
};

export default function ClearAllPointsInVideoButton({onRestart}: Props) {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const isPlaying = useAtomValue(isPlayingAtom);
  const isStreaming = useAtomValue(isStreamingAtom);
  const session = useAtomValue(sessionAtom);
  const setLabelType = useSetAtom(labelTypeAtom);
  const setLitLoRAModeEnabled = useSetAtom(litLoRAModeEnabledAtom);
  const setFrameTrackingEnabled = useSetAtom(frameTrackingEnabledAtom);
  const setLoraCandidates = useSetAtom(loraMaskCandidatesAtom);
  const setLoraTrainingData = useSetAtom(loraTrainingDataAtom);
  const setMemoryInitialized = useSetAtom(memoryInitializedAtom);
  const setSession = useSetAtom(sessionAtom);
  const setCorrectedFrames = useSetAtom(correctedFramesAtom);
  const setSessionResetKey = useSetAtom(sessionResetKeyAtom);
  const {clearMessage} = useMessagesSnackbar();
  const {restartSession} = useRestartSession();
  const {inputVideo} = useInputVideo();

  const video = useVideo();

  async function handleRestart() {
    if (video === null) {
      return;
    }

    setIsLoading(true);
    if (isPlaying) {
      video.pause();
    }
    if (isStreaming) {
      await video.abortStreamMasks();
    }
    
    // Reset LoRA states before clearing points
    try {
      await video.startOver();
    } catch (error) {
      console.error('Failed to reset LoRA states:', error);
    }
    
    // Reset BehaviorTracker in the main thread (the worker also resets its own instance)
    console.log('[ClearAllPointsInVideoButton] Resetting main thread BehaviorTracker');
    console.log('[ClearAllPointsInVideoButton] Current session:', session);
    console.log('[ClearAllPointsInVideoButton] Input video:', inputVideo);
    
    behaviorTracker.reset();
    
    // Restart the session with current session ID and video path
    if (session?.id && inputVideo?.path) {
      console.log('[ClearAllPointsInVideoButton] Restarting BehaviorTracker with session:', session.id, 'video:', inputVideo.path);
      behaviorTracker.startSession(session.id, inputVideo.path);
    } else {
      console.warn('[ClearAllPointsInVideoButton] No session ID or video path available to restart BehaviorTracker');
      console.warn('[ClearAllPointsInVideoButton] session:', session, 'inputVideo:', inputVideo);
    }
    
    const isSuccessful = await video.clearPointsInVideo();
    if (!isSuccessful) {
      await restartSession();
    }
    
    // Reset UI state atoms to default values
    video.frame = 0;
    setLabelType('positive');
    setLitLoRAModeEnabled(false);
    setFrameTrackingEnabled(false);
    setMemoryInitialized(false); // Reset memory initialization flag
    setLoraCandidates(null);
    setLoraTrainingData([]);
    setCorrectedFrames(new Set<number>()); // Reset correction counter
    setSessionResetKey(prev => prev + 1); // Trigger timer reset
    
    // Reset session ranPropagation flag
    setSession(prev => {
      if (prev === null) {
        return prev;
      }
      return {...prev, ranPropagation: false};
    });
    
    onRestart();
    clearMessage();
    setIsLoading(false);
  }

  return (
    <div {...stylex.props(styles.container)}>
      <Button
        color="ghost"
        onClick={handleRestart}
        className="!px-4 !rounded-full font-medium text-white hover:bg-black"
        startIcon={isLoading ? <Loading size="sm" /> : <Reset size={20} />}>
        Start over
      </Button>
    </div>
  );
}

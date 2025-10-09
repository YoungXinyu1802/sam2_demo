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
import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import {behaviorTracker} from '@/common/utils/BehaviorTracker';
import useVideo from '@/common/components/video/editor/useVideo';
import {
  litLoRAModeEnabledAtom,
  streamingStateAtom,
  trackletObjectsAtom,
} from '@/demo/atoms';
import {Analytics, ChartLineSmooth} from '@carbon/icons-react';
import {useAtom, useAtomValue} from 'jotai';
import {useCallback} from 'react';

export default function LITLoRAModeButton() {
  const video = useVideo();
  const {enqueueMessage} = useMessagesSnackbar();
  const streamingState = useAtomValue(streamingStateAtom);
  const tracklets = useAtomValue(trackletObjectsAtom);
  const [isLITLoRAModeEnabled, setIsLITLoRAModeEnabled] = useAtom(
    litLoRAModeEnabledAtom,
  );

  const areObjectsInitialized = tracklets.some(
    tracklet => tracklet.isInitialized,
  );

  const isDisabled =
    !areObjectsInitialized ||
    streamingState === 'requesting' ||
    streamingState === 'partial' ||
    streamingState === 'aborting';

  const handleToggleLITLoRAMode = useCallback(() => {
    if (isDisabled) {
      return;
    }

    if (!isLITLoRAModeEnabled) {
      // Enable LIT_LoRA mode
      enqueueMessage('frameTrackingEnabled'); // Reuse existing message
      video?.enableLITLoRAMode();
      setIsLITLoRAModeEnabled(true);
      behaviorTracker.logTrackingEvent('enable_frame_tracking');
    } else {
      // Disable LIT_LoRA mode
      enqueueMessage('frameTrackingDisabled'); // Reuse existing message
      video?.disableLITLoRAMode();
      setIsLITLoRAModeEnabled(false);
      behaviorTracker.logTrackingEvent('disable_frame_tracking');
    }
  }, [isLITLoRAModeEnabled, isDisabled, video, enqueueMessage, setIsLITLoRAModeEnabled]);

  return (
    <PrimaryCTAButton
      disabled={isDisabled}
      onClick={handleToggleLITLoRAMode}
      endIcon={
        isLITLoRAModeEnabled ? (
          <ChartLineSmooth size={20} />
        ) : (
          <Analytics size={20} />
        )
      }>
      {isLITLoRAModeEnabled
        ? 'Disable LIT_LoRA Mode'
        : 'Enable LIT_LoRA Mode'}
    </PrimaryCTAButton>
  );
}


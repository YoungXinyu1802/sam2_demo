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
  frameTrackingEnabledAtom,
  streamingStateAtom,
} from '@/demo/atoms';
import {Renew, Stop} from '@carbon/icons-react';
import {useAtom, useAtomValue} from 'jotai';
import {useCallback} from 'react';

export default function FrameTrackingButton() {
  const video = useVideo();
  const {enqueueMessage} = useMessagesSnackbar();
  const streamingState = useAtomValue(streamingStateAtom);
  const [isFrameTrackingEnabled, setIsFrameTrackingEnabled] = useAtom(
    frameTrackingEnabledAtom,
  );

  // Note: Frame tracking can now be enabled before objects are initialized

  const isDisabled =
    streamingState === 'requesting' ||
    streamingState === 'partial' ||
    streamingState === 'aborting';

  // Note: Frame tracking can now be enabled before objects are initialized

  const handleToggleFrameTracking = useCallback(() => {
    if (isDisabled) {
      return;
    }

    if (!isFrameTrackingEnabled) {
      // Enable frame tracking
      enqueueMessage('frameTrackingEnabled');
      video?.enableFrameTracking();
      setIsFrameTrackingEnabled(true);
      behaviorTracker.logTrackingEvent('enable_frame_tracking');
    } else {
      // Disable frame tracking
      enqueueMessage('frameTrackingDisabled');
      video?.disableFrameTracking();
      setIsFrameTrackingEnabled(false);
      behaviorTracker.logTrackingEvent('disable_frame_tracking');
    }
  }, [isFrameTrackingEnabled, isDisabled, video, enqueueMessage]);

  return (
    <PrimaryCTAButton
      disabled={isDisabled}
      onClick={handleToggleFrameTracking}
      endIcon={
        isFrameTrackingEnabled ? (
          <Stop size={20} />
        ) : (
          <Renew size={20} />
        )
      }>
      {isFrameTrackingEnabled
        ? 'Stop Frame Tracking'
        : 'Track Frame-by-Frame'}
    </PrimaryCTAButton>
  );
}


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
import useVideo from '@/common/components/video/editor/useVideo';
import {streamingStateAtom, trackletObjectsAtom} from '@/demo/atoms';
import {Renew, Stop} from '@carbon/icons-react';
import {useAtomValue} from 'jotai';
import {useCallback, useEffect, useState} from 'react';

export default function FrameTrackingButton() {
  const video = useVideo();
  const {enqueueMessage} = useMessagesSnackbar();
  const streamingState = useAtomValue(streamingStateAtom);
  const tracklets = useAtomValue(trackletObjectsAtom);
  const [isFrameTrackingEnabled, setIsFrameTrackingEnabled] = useState(false);

  const areObjectsInitialized = tracklets.some(
    tracklet => tracklet.isInitialized,
  );

  const isDisabled =
    !areObjectsInitialized ||
    streamingState === 'requesting' ||
    streamingState === 'partial' ||
    streamingState === 'aborting';

  // Reset frame tracking when switching videos or clearing points
  useEffect(() => {
    if (!areObjectsInitialized && isFrameTrackingEnabled) {
      video?.disableFrameTracking();
      setIsFrameTrackingEnabled(false);
    }
  }, [areObjectsInitialized, isFrameTrackingEnabled, video]);

  const handleToggleFrameTracking = useCallback(() => {
    if (isDisabled) {
      return;
    }

    if (!isFrameTrackingEnabled) {
      // Enable frame tracking
      enqueueMessage('frameTrackingEnabled');
      video?.enableFrameTracking();
      setIsFrameTrackingEnabled(true);
    } else {
      // Disable frame tracking
      enqueueMessage('frameTrackingDisabled');
      video?.disableFrameTracking();
      setIsFrameTrackingEnabled(false);
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


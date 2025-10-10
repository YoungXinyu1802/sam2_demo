import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import useVideo from '@/common/components/video/editor/useVideo';
import {
  litLoRAModeEnabledAtom,
  streamingStateAtom,
  trackletObjectsAtom,
} from '@/demo/atoms';
import {Checkmark} from '@carbon/icons-react';
import {useAtomValue} from 'jotai';
import {useCallback} from 'react';

export default function FinishCorrectionButton() {
  const video = useVideo();
  const {enqueueMessage} = useMessagesSnackbar();
  const streamingState = useAtomValue(streamingStateAtom);
  const tracklets = useAtomValue(trackletObjectsAtom);
  const isLITLoRAModeEnabled = useAtomValue(litLoRAModeEnabledAtom);

  const areObjectsInitialized = tracklets.some(
    tracklet => tracklet.isInitialized,
  );

  const isDisabled =
    !areObjectsInitialized ||
    !isLITLoRAModeEnabled ||
    streamingState === 'requesting' ||
    streamingState === 'partial' ||
    streamingState === 'aborting';

  const handleFinishCorrection = useCallback(() => {
    if (isDisabled) {
      return;
    }

    video?.finishCorrection();
    enqueueMessage('correctionSaved');
    // Note: Using existing 'play' event since there's no specific event for this action
  }, [isDisabled, video, enqueueMessage]);

  if (!isLITLoRAModeEnabled) {
    return null;
  }

  return (
    <PrimaryCTAButton
      disabled={isDisabled}
      onClick={handleFinishCorrection}
      endIcon={<Checkmark size={20} />}>
      Finish Correction
    </PrimaryCTAButton>
  );
}


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
import ChangeVideo from '@/common/components/gallery/ChangeVideoModal';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import useVideo from '@/common/components/video/editor/useVideo';
import {VideoData} from '@/demo/atoms';
import {DEMO_SHORT_NAME} from '@/demo/DemoConfig';
import {loadMaskImageToRLE} from '@/common/utils/MaskUtils';
import {useEffect, useRef, useState} from 'react';
import {useAtom, useAtomValue} from 'jotai';
import {
  activeTrackletObjectIdAtom,
  sessionAtom,
  trackletObjectsAtom,
} from '@/demo/atoms';
import PrimaryCTAButton from '@/common/components/button/PrimaryCTAButton';

type Props = {
  video?: VideoData;
};

export default function FirstClickView({video}: Props) {
  const isFirstClickMessageShown = useRef(false);
  const {enqueueMessage} = useMessagesSnackbar();
  const videoWorker = useVideo();
  const [session] = useAtom(sessionAtom);
  const [activeTrackletId] = useAtom(activeTrackletObjectIdAtom);
  const tracklets = useAtomValue(trackletObjectsAtom);
  const [isLoadingMask, setIsLoadingMask] = useState(false);

  useEffect(() => {
    if (!isFirstClickMessageShown.current) {
      isFirstClickMessageShown.current = true;
      enqueueMessage('firstClick');
    }
  }, [enqueueMessage]);

  const handleLoadMask = async () => {
    if (!video || !videoWorker || !session) {
      return;
    }

    // Get the first tracklet (id 0 is created automatically when session starts)
    // Use activeTrackletId if available, otherwise use the first tracklet
    const trackletId = activeTrackletId ?? (tracklets.length > 0 ? tracklets[0].id : 0);

    // Extract just the filename from the video path (remove subdirectories)
    // e.g., "gallery/555_tear_aluminium_foil.mp4" -> "555_tear_aluminium_foil.png"
    const videoFileName = video.path.split('/').pop() || video.path;
    const maskFileName = videoFileName.replace(/\.(mp4|mov|avi|mkv)$/i, '.png');
    const maskUrl = `/masks/${maskFileName}`;

    setIsLoadingMask(true);
    try {
      // Get video dimensions to resize mask to match
      const videoWidth = videoWorker.width;
      const videoHeight = videoWorker.height;
      
      console.log('[FirstClickView] Loading mask:', {
        maskUrl,
        videoWidth,
        videoHeight,
        trackletId,
        sessionId: session?.id,
      });
      
      if (videoWidth === 0 || videoHeight === 0) {
        throw new Error('Video dimensions not available');
      }
      
      // Load mask image and convert to RLE, resizing to match video dimensions
      const rleMask = await loadMaskImageToRLE(maskUrl, videoWidth, videoHeight);
      
      console.log('[FirstClickView] Mask loaded and converted to RLE:', {
        rleSize: rleMask.size,
        rleCountsLength: rleMask.counts.length,
        rleCountsPreview: rleMask.counts.slice(0, 10),
      });
      
      // Apply mask to first frame (frame 0)
      console.log('[FirstClickView] Sending mask to backend:', {
        frameIndex: 0,
        trackletId,
        rleSize: rleMask.size,
      });
      
      await videoWorker.addMask(0, trackletId, rleMask);
      
      console.log('[FirstClickView] Mask successfully applied');
      enqueueMessage('pointClick');
    } catch (error) {
      console.error('[FirstClickView] Failed to load mask:', error);
      enqueueMessage('addObjectClick');
    } finally {
      setIsLoadingMask(false);
    }
  };

  // Check if mask file exists for this video
  // Extract just the filename from the video path (remove subdirectories)
  const videoFileName = video?.path.split('/').pop() || video?.path;
  const maskFileName = videoFileName?.replace(/\.(mp4|mov|avi|mkv)$/i, '.png');
  const maskUrl = maskFileName ? `/masks/${maskFileName}` : null;

  return (
    <div className="w-full h-full flex flex-col p-8">
      <div className="grow flex flex-col gap-6">
        <h2 className="text-2xl">Click an object in the video to start</h2>
        <p className="!text-gray-60">
          You&apos;ll be able to use {DEMO_SHORT_NAME} to make fun edits to any
          video by tracking objects and applying visual effects.
        </p>
        <p className="!text-gray-60">
          To start, click any object in the video.
        </p>
        {maskUrl && session && (
          <div className="flex flex-col gap-4">
            <p className="!text-gray-60">
              Or load a pre-existing mask from the masks directory:
            </p>
            <PrimaryCTAButton
              disabled={isLoadingMask}
              onClick={handleLoadMask}>
              {isLoadingMask ? 'Loading mask...' : 'Load mask from file'}
            </PrimaryCTAButton>
          </div>
        )}
      </div>
      <div className="flex items-center">
        <ChangeVideo />
      </div>
    </div>
  );
}

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
import {loraMaskCandidatesAtom} from '@/demo/atoms';
import {useAtom} from 'jotai';
import {useCallback, useEffect} from 'react';
import useVideo from '@/common/components/video/editor/useVideo';
import Logger from '@/common/logger/Logger';

export default function LoRACandidateSelector() {
  const [candidateData, setCandidateData] = useAtom(loraMaskCandidatesAtom);
  const video = useVideo();

  const handleAcceptCandidate = useCallback(() => {
    if (!candidateData || !video) {
      return;
    }

    try {
      // Accept the first candidate (index 0)
      Logger.info('Accepting LoRA candidate (first one)');
      
      const tracker = video.getWorker_ONLY_USE_WITH_CAUTION();
      
      // Apply the first candidate
      tracker.postMessage({
        action: 'applyLoraCandidate',
        objectId: candidateData.objectId,
        frameIndex: candidateData.frameIndex,
        candidateIndex: 0,
      });

      // Clear candidates from display
      setCandidateData(null);
    } catch (error) {
      Logger.error('Failed to apply LoRA candidate:', error);
    }
  }, [candidateData, video, setCandidateData]);

  const handleRejectCandidate = useCallback(() => {
    if (!video) {
      return;
    }

    Logger.info('Rejected LoRA candidate');
    
    // Clear candidates from video overlay
    const tracker = video.getWorker_ONLY_USE_WITH_CAUTION();
    tracker.postMessage({
      action: 'clearLoraCandidates',
    });
    
    setCandidateData(null);
  }, [video, setCandidateData]);

  // Keyboard event handler
  useEffect(() => {
    if (!candidateData || candidateData.candidates.length === 0) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        handleAcceptCandidate();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        handleRejectCandidate();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [candidateData, handleAcceptCandidate, handleRejectCandidate]);

  if (!candidateData || candidateData.candidates.length === 0) {
    return null;
  }

  return null;
}


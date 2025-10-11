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
import {useCallback} from 'react';
import useVideo from '@/common/components/video/editor/useVideo';
import Logger from '@/common/logger/Logger';

const CANDIDATE_COLORS = [
  '#FF6B6B', // Red
  '#4ECDC4', // Cyan
  '#45B7D1', // Blue
  '#FFA07A', // Light Salmon
  '#98D8C8', // Mint
  '#F7DC6F', // Yellow
  '#BB8FCE', // Purple
  '#85C1E2', // Sky Blue
];

export default function LoRACandidateSelector() {
  const [candidateData, setCandidateData] = useAtom(loraMaskCandidatesAtom);
  const video = useVideo();

  const handleSelectCandidate = useCallback(async (index: number) => {
    if (!candidateData || !video) {
      return;
    }

    try {
      Logger.info(`Applying LoRA candidate ${index}`);
      
      // Call the tracker method to apply the candidate
      const tracker = video.getWorker_ONLY_USE_WITH_CAUTION();
      
      // Post message to worker to apply candidate
      tracker.postMessage({
        action: 'applyLoraCandidate',
        objectId: candidateData.objectId,
        frameIndex: candidateData.frameIndex,
        candidateIndex: index,
      });

      // Clear candidates from display
      setCandidateData(null);
    } catch (error) {
      Logger.error('Failed to apply LoRA candidate:', error);
    }
  }, [candidateData, video, setCandidateData]);

  const handleRejectAll = useCallback(() => {
    Logger.info('Rejected all LoRA candidates');
    setCandidateData(null);
  }, [setCandidateData]);

  if (!candidateData || candidateData.candidates.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: 'absolute',
        bottom: '100px',
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: '20px',
        borderRadius: '12px',
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        minWidth: '300px',
      }}>
      <div
        style={{
          color: 'white',
          fontSize: '16px',
          fontWeight: 'bold',
          marginBottom: '4px',
          textAlign: 'center',
        }}>
        Select a LoRA Candidate
      </div>
      <div
        style={{
          color: '#aaa',
          fontSize: '12px',
          marginBottom: '12px',
          textAlign: 'center',
        }}>
        Masks are shown on the video - click a button to apply
      </div>
      
      <div style={{display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center'}}>
        {candidateData.candidates.map((candidate, idx) => (
          <button
            key={candidate.index}
            onClick={() => handleSelectCandidate(candidate.index)}
            style={{
              padding: '12px 20px',
              backgroundColor: CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length],
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'transform 0.2s, box-shadow 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.05)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
              e.currentTarget.style.boxShadow = 'none';
            }}>
            <div
              style={{
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                backgroundColor: 'rgba(255,255,255,0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '12px',
              }}>
              {idx + 1}
            </div>
            Candidate {idx + 1}
          </button>
        ))}
      </div>

      <button
        onClick={handleRejectAll}
        style={{
          padding: '10px 20px',
          backgroundColor: '#666',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          fontSize: '14px',
          marginTop: '4px',
          transition: 'background-color 0.2s',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#555';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = '#666';
        }}>
        Reject All
      </button>
    </div>
  );
}


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
import {useAtomValue} from 'jotai';

export default function LoRACandidatePrompt() {
  const candidateData = useAtomValue(loraMaskCandidatesAtom);

  if (!candidateData || candidateData.candidates.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        position: 'absolute',
        top: '16px',
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        padding: '12px 24px',
        borderRadius: '8px',
        border: '2px solid #4ECDC4',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        zIndex: 1000,
        pointerEvents: 'none',
      }}>
      <div
        style={{
          color: '#4ECDC4',
          fontSize: '16px',
          fontWeight: 'bold',
        }}>
        Accept LoRA candidate?
      </div>
      <div
        style={{
          color: '#fff',
          fontSize: '14px',
          display: 'flex',
          gap: '8px',
          alignItems: 'center',
        }}>
        <kbd
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            padding: '4px 8px',
            borderRadius: '4px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            fontSize: '12px',
            fontWeight: 'bold',
          }}>
          Enter
        </kbd>
        <span style={{color: '#aaa'}}>to accept</span>
        <span style={{color: '#555', margin: '0 4px'}}>|</span>
        <kbd
          style={{
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            padding: '4px 8px',
            borderRadius: '4px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            fontSize: '12px',
            fontWeight: 'bold',
          }}>
          Esc
        </kbd>
        <span style={{color: '#aaa'}}>to reject</span>
      </div>
    </div>
  );
}

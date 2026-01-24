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
import ClearAllPointsInVideoButton from '@/common/components/annotations/ClearAllPointsInVideoButton';
import CloseSessionButton from '@/common/components/annotations/CloseSessionButton';
import ExportBehaviorDataButton from '@/common/components/button/ExportBehaviorDataButton';
import FrameTrackingButton from '@/common/components/button/FrameTrackingButton';
import TrackAndPlayButton from '@/common/components/button/TrackAndPlayButton';
import FPSInputBox from '@/common/components/input/FPSInputBox';
import ToolbarBottomActionsWrapper from '@/common/components/toolbar/ToolbarBottomActionsWrapper';
import {
  EFFECT_TOOLBAR_INDEX,
  OBJECT_TOOLBAR_INDEX,
} from '@/common/components/toolbar/ToolbarConfig';
import {sessionAtom, streamingStateAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';

type Props = {
  onTabChange: (newIndex: number) => void;
};

export default function ObjectsToolbarBottomActions({onTabChange}: Props) {
  const streamingState = useAtomValue(streamingStateAtom);
  const session = useAtomValue(sessionAtom);

  const isTrackingInProgress =
    streamingState !== 'none' && streamingState !== 'full';

  const isTrackingComplete = streamingState === 'full';
  
  // Show tracking buttons when there's an active session (not just when streaming)
  const hasActiveSession = session !== null;

  function handleSwitchToEffectsTab() {
    onTabChange(EFFECT_TOOLBAR_INDEX);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* FPS input in its own row above the buttons */}
      {hasActiveSession && (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center',
          paddingTop: '8px',
          paddingBottom: '4px'
        }}>
          <FPSInputBox />
        </div>
      )}
      {/* Start over button in its own row */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center',
        paddingBottom: '4px'
      }}>
        <ClearAllPointsInVideoButton
          onRestart={() => onTabChange(OBJECT_TOOLBAR_INDEX)}
        />
      </div>
      {/* Main action buttons row */}
      <ToolbarBottomActionsWrapper>
        {/* Show tracking buttons when there's an active session */}
        {hasActiveSession && (
          <>
            {isTrackingInProgress && <TrackAndPlayButton />}
            <FrameTrackingButton />
          </>
        )}
        {isTrackingComplete && (
          <>
            <ExportBehaviorDataButton />
            <CloseSessionButton onSessionClose={handleSwitchToEffectsTab} />
          </>
        )}
      </ToolbarBottomActionsWrapper>
    </div>
  );
}

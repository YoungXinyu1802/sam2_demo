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
import {behaviorTracker} from '@/common/utils/BehaviorTracker';
import {litLoRAModeEnabledAtom} from '@/demo/atoms';
import {Download} from '@carbon/icons-react';
import {useStore} from 'jotai';
import {useCallback} from 'react';

export default function ExportBehaviorDataButton() {
  const store = useStore();

  const handleExport = useCallback(() => {
    // Read the current LIT status directly from the store at export time
    // to avoid any closure or timing issues
    const isLITLoRAModeEnabled = store.get(litLoRAModeEnabledAtom);
    
    // End the session timing
    behaviorTracker.endSession();
    
    // Download the data with LIT status
    behaviorTracker.downloadData(undefined, isLITLoRAModeEnabled);
  }, [store]);

  return (
    <PrimaryCTAButton
      onClick={handleExport}
      endIcon={<Download size={20} />}>
      Export Behavior Data
    </PrimaryCTAButton>
  );
}


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
import {Download} from '@carbon/icons-react';
import {useCallback} from 'react';

export default function ExportBehaviorDataButton() {
  const handleExport = useCallback(() => {
    // End the session timing
    behaviorTracker.endSession();
    
    // Download the data
    behaviorTracker.downloadData();
  }, []);

  return (
    <PrimaryCTAButton
      onClick={handleExport}
      endIcon={<Download size={20} />}>
      Export Behavior Data
    </PrimaryCTAButton>
  );
}


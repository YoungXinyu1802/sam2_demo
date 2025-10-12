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
import {trackingFpsAtom} from '@/demo/atoms';
import useVideo from '@/common/components/video/editor/useVideo';
import {useAtom} from 'jotai';
import {useCallback} from 'react';

type Props = {
  className?: string;
};

export default function FPSInputBox({className}: Props) {
  const video = useVideo();
  const [trackingFps, setTrackingFps] = useAtom(trackingFpsAtom);

  const handleFpsChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const newFps = parseInt(event.target.value, 10);
      
      // Validate the input
      if (newFps > 0 && newFps <= 60) {
        setTrackingFps(newFps);
        video?.setTrackingFps(newFps);
      }
    },
    [setTrackingFps, video],
  );

  return (
    <div className={`flex items-center gap-2 ${className || ''}`}>
      <label htmlFor="fps-input" className="text-sm text-gray-300 whitespace-nowrap">
        Tracking FPS:
      </label>
      <input
        id="fps-input"
        type="number"
        min="1"
        max="60"
        value={trackingFps}
        onChange={handleFpsChange}
        className="w-16 px-2 py-1 text-sm bg-gray-700 border border-gray-600 rounded text-white focus:outline-none focus:border-blue-500"
        title="Set the frame rate for frame-by-frame tracking (1-60 FPS)"
      />
    </div>
  );
}

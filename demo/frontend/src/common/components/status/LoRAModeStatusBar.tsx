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
import {litLoRAModeEnabledAtom} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtomValue} from 'jotai';

const styles = stylex.create({
  statusBar: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    height: '28px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '12px',
    fontWeight: 500,
    zIndex: 1000,
    transition: 'all 0.2s ease',
  },
  enabled: {
    backgroundColor: '#10B981',
    color: '#FFFFFF',
  },
  disabled: {
    backgroundColor: '#6B7280',
    color: '#F3F4F6',
  },
  indicator: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    marginRight: '8px',
  },
  indicatorEnabled: {
    backgroundColor: '#FFFFFF',
  },
  indicatorDisabled: {
    backgroundColor: '#D1D5DB',
  },
});

export default function LoRAModeStatusBar() {
  const isLITLoRAModeEnabled = useAtomValue(litLoRAModeEnabledAtom);

  return (
    <div
      {...stylex.props(
        styles.statusBar,
        isLITLoRAModeEnabled ? styles.enabled : styles.disabled,
      )}>
      <div
        {...stylex.props(
          styles.indicator,
          isLITLoRAModeEnabled ? styles.indicatorEnabled : styles.indicatorDisabled,
        )}
      />
      <span>
        LIT-LoRA Mode: {isLITLoRAModeEnabled ? 'Enabled' : 'Disabled'}
      </span>
    </div>
  );
}

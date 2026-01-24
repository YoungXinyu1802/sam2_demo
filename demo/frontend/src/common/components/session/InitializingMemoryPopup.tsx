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
import {isInitializingMemoryAtom} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtomValue} from 'jotai';

const styles = stylex.create({
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10000,
    backdropFilter: 'blur(4px)',
  },
  popup: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: '32px 48px',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 16,
    minWidth: 300,
  },
  spinner: {
    width: 48,
    height: 48,
    border: '4px solid #f3f3f3',
    borderTop: '4px solid #3498db',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  title: {
    fontSize: 20,
    fontWeight: 600,
    color: '#333',
    margin: 0,
    textAlign: 'center',
  },
  message: {
    fontSize: 14,
    color: '#666',
    margin: 0,
    textAlign: 'center',
  },
});

// Add keyframes for spinner animation
const spinKeyframes = `
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
`;

export default function InitializingMemoryPopup() {
  const isInitializing = useAtomValue(isInitializingMemoryAtom);

  if (!isInitializing) {
    return null;
  }

  return (
    <>
      <style>{spinKeyframes}</style>
      <div {...stylex.props(styles.overlay)}>
        <div {...stylex.props(styles.popup)}>
          <div {...stylex.props(styles.spinner)} />
          <h3 {...stylex.props(styles.title)}>Initializing</h3>
          <p {...stylex.props(styles.message)}>
            Processing your annotations for tracking...
          </p>
        </div>
      </div>
    </>
  );
}

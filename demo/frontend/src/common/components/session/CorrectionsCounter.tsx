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
import {correctionsCountAtom} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtomValue} from 'jotai';

const styles = stylex.create({
  container: {
    position: 'fixed',
    top: 10,
    left: 180,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    color: '#fff',
    padding: '8px 16px',
    borderRadius: 8,
    fontSize: 14,
    fontFamily: 'monospace',
    fontWeight: 500,
    zIndex: 9999,
    backdropFilter: 'blur(4px)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
    userSelect: 'none',
  },
  label: {
    opacity: 0.8,
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    marginBottom: 2,
  },
  count: {
    fontSize: 24,
    fontWeight: 700,
    letterSpacing: '1px',
    textAlign: 'center',
  },
});

export default function CorrectionsCounter() {
  const correctionsCount = useAtomValue(correctionsCountAtom);

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.label)}>Corrections</div>
      <div {...stylex.props(styles.count)}>{correctionsCount}</div>
    </div>
  );
}

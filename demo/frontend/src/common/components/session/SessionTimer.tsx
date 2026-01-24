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
import {frameTrackingEnabledAtom, isInitializingMemoryAtom, memoryInitializedAtom, sessionResetKeyAtom} from '@/demo/atoms';
import stylex from '@stylexjs/stylex';
import {useAtomValue} from 'jotai';
import {useEffect, useRef, useState} from 'react';

const styles = stylex.create({
  container: {
    position: 'fixed',
    top: 10,
    left: 10,
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
  time: {
    fontSize: 18,
    fontWeight: 600,
    letterSpacing: '1px',
  },
  status: {
    fontSize: 10,
    opacity: 0.6,
    marginTop: 2,
  },
});

export default function SessionTimer() {
  const [elapsedTime, setElapsedTime] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const accumulatedTimeRef = useRef<number>(0);
  const isFrameTrackingEnabled = useAtomValue(frameTrackingEnabledAtom);
  const isInitializingMemory = useAtomValue(isInitializingMemoryAtom);
  const memoryInitialized = useAtomValue(memoryInitializedAtom);
  const sessionResetKey = useAtomValue(sessionResetKeyAtom);

  // Reset timer when session is reset
  useEffect(() => {
    if (sessionResetKey > 0) {
      console.log(`[SessionTimer] Resetting timer due to session reset (key: ${sessionResetKey})`);
      startTimeRef.current = null;
      accumulatedTimeRef.current = 0;
      setElapsedTime(0);
    }
  }, [sessionResetKey]);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;

    // Only track time when:
    // 1. Frame tracking is enabled
    // 2. Memory has been initialized (at least once)
    // 3. Memory is NOT currently initializing
    const shouldTrackTime = isFrameTrackingEnabled && memoryInitialized && !isInitializingMemory;

    console.log(`[SessionTimer] Effect triggered - isFrameTrackingEnabled: ${isFrameTrackingEnabled}, memoryInitialized: ${memoryInitialized}, isInitializingMemory: ${isInitializingMemory}, shouldTrackTime: ${shouldTrackTime}`);

    if (shouldTrackTime) {
      // Start tracking time
      if (startTimeRef.current === null) {
        const now = Date.now();
        startTimeRef.current = now;
        const timestamp = new Date(now).toISOString();
        console.log(`[SessionTimer] ⏱️ Timer STARTED at ${timestamp} (${now})`);
        console.log(`[SessionTimer] Initialization completed - tracking begins now!`);
      } else {
        console.log(`[SessionTimer] Resuming timer (already had start time)`);
      }

      interval = setInterval(() => {
        if (startTimeRef.current !== null) {
          const currentElapsed = Date.now() - startTimeRef.current;
          setElapsedTime(accumulatedTimeRef.current + currentElapsed);
        }
      }, 100); // Update more frequently for better UX
    } else {
      // Stop tracking time
      if (startTimeRef.current !== null) {
        const currentElapsed = Date.now() - startTimeRef.current;
        accumulatedTimeRef.current += currentElapsed;
        setElapsedTime(accumulatedTimeRef.current);
        startTimeRef.current = null;
        console.log(`[SessionTimer] Pausing timer - accumulated time: ${accumulatedTimeRef.current}ms`);
      } else {
        console.log(`[SessionTimer] Timer not started yet, nothing to pause`);
      }
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isFrameTrackingEnabled, memoryInitialized, isInitializingMemory]);

  const formatTime = (milliseconds: number): string => {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  // Determine the status message
  const getStatus = () => {
    if (!isFrameTrackingEnabled) {
      return '○ Paused';
    }
    if (!memoryInitialized || isInitializingMemory) {
      return '⏸ Waiting...';
    }
    return '● Recording';
  };

  return (
    <div {...stylex.props(styles.container)}>
      <div {...stylex.props(styles.label)}>Session Time</div>
      <div {...stylex.props(styles.time)}>{formatTime(elapsedTime)}</div>
      <div {...stylex.props(styles.status)}>
        {getStatus()}
      </div>
    </div>
  );
}

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
import {decode} from '@/jscocotools/mask';
import {useAtomValue} from 'jotai';
import {useEffect, useRef} from 'react';
import type {RLEObject} from '@/jscocotools/mask';

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

type Props = {
  width: number;
  height: number;
};

export function LoRACandidatesLayer({width, height}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const candidateData = useAtomValue(loraMaskCandidatesAtom);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!candidateData || candidateData.candidates.length === 0) {
      return;
    }

    // Render each candidate mask with its color
    candidateData.candidates.forEach((candidate, idx) => {
      try {
        // candidate.mask is the RLE-encoded mask data from backend
        const maskRLE = candidate.mask as any;
        const rleObject: RLEObject = {
          size: maskRLE.size as [number, number],
          counts: maskRLE.counts,
        };

        // Decode the RLE mask
        const decodedMask = decode([rleObject]);
        const maskData = decodedMask.data;
        const [maskHeight, maskWidth] = decodedMask.shape;

        // Create an ImageData for this candidate
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        // Parse the color
        const color = CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length];
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);

        // Fill the mask pixels with the color (semi-transparent)
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            // Map canvas coordinates to mask coordinates
            const maskX = Math.floor((x / width) * maskWidth);
            const maskY = Math.floor((y / height) * maskHeight);
            const maskIdx = maskY * maskWidth + maskX;

            if (maskData[maskIdx] > 0) {
              const pixelIdx = (y * width + x) * 4;
              data[pixelIdx] = r;
              data[pixelIdx + 1] = g;
              data[pixelIdx + 2] = b;
              data[pixelIdx + 3] = 120; // Semi-transparent (47% opacity)
            }
          }
        }

        // Draw the mask
        ctx.putImageData(imageData, 0, 0);

        // Draw candidate number on the mask (find centroid)
        let sumX = 0;
        let sumY = 0;
        let count = 0;

        for (let y = 0; y < maskHeight; y++) {
          for (let x = 0; x < maskWidth; x++) {
            const maskIdx = y * maskWidth + x;
            if (maskData[maskIdx] > 0) {
              sumX += x;
              sumY += y;
              count++;
            }
          }
        }

        if (count > 0) {
          const centroidX = Math.floor((sumX / count / maskWidth) * width);
          const centroidY = Math.floor((sumY / count / maskHeight) * height);

          // Draw number badge
          ctx.save();
          
          // Draw white circle background
          ctx.fillStyle = 'white';
          ctx.beginPath();
          ctx.arc(centroidX, centroidY, 20, 0, 2 * Math.PI);
          ctx.fill();

          // Draw colored border
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.arc(centroidX, centroidY, 20, 0, 2 * Math.PI);
          ctx.stroke();

          // Draw number
          ctx.fillStyle = color;
          ctx.font = 'bold 20px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText((idx + 1).toString(), centroidX, centroidY);

          ctx.restore();
        }
      } catch (error) {
        console.error(`Failed to render candidate ${idx}:`, error);
      }
    });
  }, [candidateData, width, height]);

  if (!candidateData || candidateData.candidates.length === 0) {
    return null;
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 100,
      }}
    />
  );
}


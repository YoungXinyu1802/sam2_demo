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
        // It has structure: { size: [height, width], counts: string }
        const maskRLE = candidate.mask as any;
        
        // Create properly formatted RLEObject
        const rleObject: RLEObject = {
          size: [maskRLE.size[0], maskRLE.size[1]] as [number, number],
          counts: maskRLE.counts,
        };

        // Decode the RLE mask to get binary mask data
        const decodedMask = decode([rleObject]);
        const maskData = decodedMask.data as Uint8Array;
        const [maskHeight, maskWidth] = decodedMask.shape;
        
        console.log(`[LoRA Candidate ${idx}]`);
        console.log(`  - RLE size from backend: [${maskRLE.size[0]}, ${maskRLE.size[1]}]`);
        console.log(`  - Decoded shape: [${maskHeight}, ${maskWidth}]`);
        console.log(`  - Canvas (width x height): ${width} x ${height}`);

        // Parse the color
        const color = CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length];
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);

        // RLE masks are stored in column-major (Fortran) order.
        // The decoded mask has shape [height, width] but the data array is in column-major format.
        // This means: for pixel at (row y, col x) in the image, the index is: x * height + y
        // We need to convert to row-major format for ImageData which expects: y * width + x
        
        const imageData = ctx.createImageData(maskWidth, maskHeight);
        const data = imageData.data;

        // Convert from column-major (x * h + y) to row-major (y * w + x)
        for (let y = 0; y < maskHeight; y++) {
          for (let x = 0; x < maskWidth; x++) {
            // Column-major index (how maskData is stored)
            const srcIdx = x * maskHeight + y;
            // Row-major index (how ImageData expects it)
            const dstIdx = (y * maskWidth + x) * 4;
            
            if (maskData[srcIdx] > 0) {
              data[dstIdx] = r;
              data[dstIdx + 1] = g;
              data[dstIdx + 2] = b;
              data[dstIdx + 3] = 120; // Semi-transparent
            }
          }
        }

        // Draw to canvas - scale if dimensions don't match
        if (maskWidth !== width || maskHeight !== height) {
          const tempCanvas = new OffscreenCanvas(maskWidth, maskHeight);
          const tempCtx = tempCanvas.getContext('2d');
          if (tempCtx) {
            tempCtx.putImageData(imageData, 0, 0);
            ctx.drawImage(tempCanvas, 0, 0, width, height);
          }
        } else {
          ctx.putImageData(imageData, 0, 0);
        }

        // Calculate centroid for the number badge
        let sumX = 0;
        let sumY = 0;
        let count = 0;

        for (let y = 0; y < maskHeight; y++) {
          for (let x = 0; x < maskWidth; x++) {
            const srcIdx = x * maskHeight + y;
            if (maskData[srcIdx] > 0) {
              sumX += x;
              sumY += y;
              count++;
            }
          }
        }

        if (count > 0) {
          // Calculate centroid and scale to canvas coordinates
          const maskCentroidX = sumX / count;
          const maskCentroidY = sumY / count;
          const scaleX = width / maskWidth;
          const scaleY = height / maskHeight;
          const centroidX = Math.floor(maskCentroidX * scaleX);
          const centroidY = Math.floor(maskCentroidY * scaleY);

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


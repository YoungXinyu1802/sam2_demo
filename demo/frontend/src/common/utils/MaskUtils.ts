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
import {DataArray, RLEObject, encode} from '@/jscocotools/mask';

/**
 * Converts an image mask represented as a binary image (foreground pixels are
 * `>1` and background pixels are `0`) stored in a Uint8Array to an RGBA
 * representation where background pixels have an alpha value of 0 and
 * foreground pixels have an alpha value of 255. This is useful for compositing
 * the mask onto another image.
 *
 * ```typescript
 * const rgba = convertMaskDataToRGBA(mask.data);
 * ```
 *
 * @param data - The image mask represented as a Uint8Array
 * @returns A new Uint8ClampedArray representing the mask in RGBA format
 */
export function convertMaskToRGBA(data: Uint8Array): Uint8ClampedArray {
  // Shifting pixels instead of assigning them individually per pixel is
  // much faster. See JSPerf benchamrk: https://jsperf.app/morifo
  const len = data.length;
  const tempData = new Uint32Array(len);
  const RGA = 0x00ffffff;
  const FOREGROUND = 0xff000000;
  const BACKGROUND = 0x00000000;
  for (let i = 0; i < len; i++) {
    const alpha = data[i] > 0 ? FOREGROUND : BACKGROUND; // alpha is the high byte. Bits 24-31
    tempData[i] = alpha + RGA;
  }
  return new Uint8ClampedArray(tempData.buffer);
}

/**
 * Loads a mask image from a URL and converts it to RLE format.
 * The mask image should be a binary image where foreground pixels are non-zero
 * and background pixels are zero.
 * If targetWidth and targetHeight are provided, the mask will be resized to match
 * these dimensions before encoding.
 *
 * @param imageUrl - URL of the mask image to load
 * @param targetWidth - Optional target width to resize the mask to (should match video width)
 * @param targetHeight - Optional target height to resize the mask to (should match video height)
 * @returns Promise that resolves to an RLEObject representing the mask
 */
export async function loadMaskImageToRLE(
  imageUrl: string,
  targetWidth?: number,
  targetHeight?: number,
): Promise<RLEObject> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      try {
        // Use target dimensions if provided, otherwise use image dimensions
        const width = targetWidth ?? img.width;
        const height = targetHeight ?? img.height;
        
        console.log('[MaskUtils] Loading mask image:', {
          imageUrl,
          originalSize: {width: img.width, height: img.height},
          targetSize: {width, height},
        });
        
        // Create a canvas with target dimensions to extract and resize pixel data
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }
        
        // Draw and resize the image to match target dimensions
        ctx.drawImage(img, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);
        
        // Convert RGBA to binary mask (select areas where pixel value > 0)
        // For grayscale masks, R=G=B, so we can check any single channel
        // This matches Python's img > 0 behavior where img is grayscale (height, width)
        
        // First, extract mask data in row-major order (standard JavaScript array order)
        const rowMajorData = new Uint8Array(width * height);
        let foregroundPixels = 0;
        let backgroundPixels = 0;
        for (let i = 0; i < imageData.data.length; i += 4) {
          // For grayscale images, R=G=B, so check red channel (or any channel)
          // Canvas getImageData always returns RGBA format
          const grayValue = imageData.data[i]; // R channel (same as G and B for grayscale)
          const pixelIndex = i / 4;
          
          // Simple check: if pixel value > 0, it's foreground
          // This matches Python's: mask = img > 0 where img is grayscale
          const isForeground = grayValue > 0;
          rowMajorData[pixelIndex] = isForeground ? 255 : 0;
          if (isForeground) {
            foregroundPixels++;
          } else {
            backgroundPixels++;
          }
        }
        
        console.log('[MaskUtils] Mask statistics:', {
          totalPixels: width * height,
          foregroundPixels,
          backgroundPixels,
          foregroundPercentage: ((foregroundPixels / (width * height)) * 100).toFixed(2) + '%',
        });
        
        // Convert from row-major (y * width + x) to column-major (x * height + y) for RLE encoding
        // RLE masks are stored in column-major (Fortran) order
        // For pixel at (row y, col x): row-major index = y * width + x, column-major index = x * height + y
        const maskData = new Uint8Array(width * height);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const rowMajorIndex = y * width + x;
            const colMajorIndex = x * height + y;
            maskData[colMajorIndex] = rowMajorData[rowMajorIndex];
          }
        }
        
        // Convert to DataArray format expected by encode
        // Shape is [height, width, 1] and data is in column-major order
        const dataArray = new DataArray(maskData, [height, width, 1]);
        
        // Encode to RLE
        const rleObjects = encode(dataArray);
        if (rleObjects.length === 0) {
          reject(new Error('Failed to encode mask'));
          return;
        }
        
        const rleMask = rleObjects[0];
        console.log('[MaskUtils] RLE encoding complete:', {
          rleSize: rleMask.size,
          rleCountsLength: rleMask.counts.length,
        });
        
        resolve(rleMask);
      } catch (error) {
        console.error('[MaskUtils] Error processing mask:', error);
        reject(error);
      }
    };
    
    img.onerror = () => {
      reject(new Error(`Failed to load mask image from ${imageUrl}`));
    };
    
    img.src = imageUrl;
  });
}

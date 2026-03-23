# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mllama-specific multimodal preprocessing utilities."""

import numpy as np
from PIL import Image

from maxtext.multimodal import utils as mm_utils

MLLAMA_IMAGE_SIZE = 560
MLLAMA_MAX_NUM_TILES = 4
MLLAMA_IMAGE_MEAN = (0.5,) * 3
MLLAMA_IMAGE_STD = (0.5,) * 3
MLLAMA_PIXEL_VALUE_RESCALE_FACTOR = 1.0 / 255.0
MLLAMA_SUPPORTED_ASPECT_RATIOS = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1))


def _find_best_ratio(height: int, width: int) -> tuple[int, int]:
  target_ratio = height / width
  return min(MLLAMA_SUPPORTED_ASPECT_RATIOS, key=lambda ratio: abs((ratio[0] / ratio[1]) - target_ratio))


def _resize_to_canvas(image: np.ndarray, aspect_ratio: tuple[int, int]) -> np.ndarray:
  target_h = aspect_ratio[0] * MLLAMA_IMAGE_SIZE
  target_w = aspect_ratio[1] * MLLAMA_IMAGE_SIZE
  pil_image = Image.fromarray(image)
  resized = pil_image.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
  return np.asarray(resized)


def _split_to_tiles(image: np.ndarray, aspect_ratio: tuple[int, int]) -> np.ndarray:
  ratio_h, ratio_w = aspect_ratio
  tile_h = image.shape[0] // ratio_h
  tile_w = image.shape[1] // ratio_w
  tiles = []
  for row in range(ratio_h):
    for col in range(ratio_w):
      tile = image[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w]
      tiles.append(np.transpose(tile, (2, 0, 1)))
  while len(tiles) < MLLAMA_MAX_NUM_TILES:
    tiles.append(np.zeros((mm_utils.NUM_IMAGE_CHANNELS, MLLAMA_IMAGE_SIZE, MLLAMA_IMAGE_SIZE), dtype=np.float32))
  return np.stack(tiles, axis=0)


def preprocess_mm_data_mllama(images):
  """Tile and normalize images for Mllama / Llama 3.2 Vision."""
  images_in = [images] if isinstance(images, np.ndarray) else list(images)
  image_tiles, image_masks, aspect_ratios = [], [], []

  for image in images_in:
    aspect_ratio = _find_best_ratio(image.shape[0], image.shape[1])
    resized = _resize_to_canvas(image, aspect_ratio).astype(np.float32)
    normalized = mm_utils.normalize_images(
        resized * MLLAMA_PIXEL_VALUE_RESCALE_FACTOR,
        mean=MLLAMA_IMAGE_MEAN,
        std=MLLAMA_IMAGE_STD,
    )
    image_tiles.append(_split_to_tiles(normalized, aspect_ratio))

    mask = np.zeros((MLLAMA_MAX_NUM_TILES,), dtype=np.int32)
    mask[: aspect_ratio[0] * aspect_ratio[1]] = 1
    image_masks.append(mask)
    aspect_ratios.append(aspect_ratio)

  return mm_utils.PreprocessorOutput(
      pixel_values=np.stack(image_tiles, axis=0).astype(np.float32),
      pixel_mask=np.stack(image_masks, axis=0).astype(np.int32),
      aspect_ratios=np.asarray(aspect_ratios, dtype=np.int32),
      num_images=len(images_in),
  )


def reformat_prompt_mllama(prompt, image_placeholder, num_images):
  if image_placeholder in prompt:
    prompt = prompt.replace(image_placeholder, "<|image|>")
  placeholder_count = prompt.count("<|image|>")
  if placeholder_count < num_images:
    prompt = "<|image|>" * (num_images - placeholder_count) + prompt
  return prompt


def get_dummy_image_shape_for_init_mllama(batch_size=1, num_image_per_sequence=1):
  return (
      batch_size * num_image_per_sequence,
      MLLAMA_MAX_NUM_TILES,
      mm_utils.NUM_IMAGE_CHANNELS,
      MLLAMA_IMAGE_SIZE,
      MLLAMA_IMAGE_SIZE,
  )

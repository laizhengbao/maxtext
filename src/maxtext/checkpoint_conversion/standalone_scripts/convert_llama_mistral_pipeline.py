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

"""Unified CLI for Meta/PyTorch-style checkpoint conversion workflows.

This wrapper reuses the existing converter entrypoints instead of reimplementing
model-specific logic. It always converts an incoming checkpoint into MaxText's
Orbax format and can optionally chain that output into the generic
``to_huggingface`` exporter when the target model is already supported there.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from maxtext.utils.globals import HF_IDS, MAXTEXT_CONFIGS_DIR

_MAXTEXT_CONVERTER_MODULE = "maxtext.checkpoint_conversion.standalone_scripts.llama_or_mistral_ckpt"
_TO_HF_MODULE = "maxtext.checkpoint_conversion.to_huggingface"
_EXPORT_SUPPORTED_MODELS = frozenset(model_name for model_name in HF_IDS if model_name != "default")
_EXAMPLE_EXPORT_SUPPORTED_MODELS = ("llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "mixtral-8x7b", "mixtral-8x22b")


def _build_to_maxtext_command(args: argparse.Namespace) -> list[str]:
  command = [
      sys.executable,
      "-m",
      _MAXTEXT_CONVERTER_MODULE,
      "--base-model-path",
      args.base_model_path,
      "--maxtext-model-path",
      args.maxtext_model_path,
      "--model-size",
      args.model_size,
      "--huggingface-checkpoint",
      str(args.huggingface_checkpoint).lower(),
      "--use-ocdbt",
      str(args.use_ocdbt).lower(),
      "--use-zarr3",
      str(args.use_zarr3).lower(),
  ]
  if args.lora_input_adapters_path:
    command.extend(["--lora-input-adapters-path", args.lora_input_adapters_path])
  return command


def _build_to_huggingface_command(args: argparse.Namespace) -> list[str]:
  if args.model_size not in _EXPORT_SUPPORTED_MODELS:
    supported_models = ", ".join(
        sorted(model for model in _EXPORT_SUPPORTED_MODELS if model in _EXAMPLE_EXPORT_SUPPORTED_MODELS)
    )
    raise ValueError(
        "`--hf-model-path` is only supported for model sizes handled by "
        f"`maxtext.checkpoint_conversion.to_huggingface`. Got {args.model_size!r}. "
        f"Examples of supported model sizes: {supported_models}."
    )

  config_path = args.config or str(Path(MAXTEXT_CONFIGS_DIR) / "base.yml")
  checkpoint_path = f"{args.maxtext_model_path.rstrip('/')}/0/items"
  command = [
      sys.executable,
      "-m",
      _TO_HF_MODULE,
      config_path,
      f"model_name={args.model_size}",
      f"load_parameters_path={checkpoint_path}",
      f"base_output_directory={args.hf_model_path}",
      f"scan_layers={str(args.scan_layers).lower()}",
      f"weight_dtype={args.weight_dtype}",
      "use_multimodal=false",
  ]
  if args.hf_access_token:
    command.append(f"hf_access_token={args.hf_access_token}")
  return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--base-model-path", required=True, help="Path to the source Meta/PyTorch-style checkpoint.")
  parser.add_argument("--maxtext-model-path", required=True, help="Destination directory for the converted MaxText checkpoint.")
  parser.add_argument("--model-size", required=True, help="Model key understood by llama_or_mistral_ckpt, e.g. llama3.1-8b.")
  parser.add_argument(
      "--huggingface-checkpoint",
      action="store_true",
      help="Interpret the source checkpoint as Hugging Face safetensors rather than Meta .pth shards.",
  )
  parser.add_argument("--lora-input-adapters-path", default="", help="Optional directory containing LoRA adapters to convert.")
  parser.add_argument("--use-ocdbt", action=argparse.BooleanOptionalAction, default=True, help="Forwarded to the MaxText converter.")
  parser.add_argument("--use-zarr3", action=argparse.BooleanOptionalAction, default=True, help="Forwarded to the MaxText converter.")
  parser.add_argument(
      "--hf-model-path",
      default="",
      help="Optional output directory or remote destination for a chained MaxText→HuggingFace export.",
  )
  parser.add_argument(
      "--config",
      default="",
      help="Optional MaxText config path for the chained HuggingFace export. Defaults to src/maxtext/configs/base.yml.",
  )
  parser.add_argument(
      "--scan-layers",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Whether the intermediate MaxText checkpoint uses scanned layers for the chained HuggingFace export.",
  )
  parser.add_argument(
      "--weight-dtype",
      default="bfloat16",
      help="Weight dtype forwarded to maxtext.checkpoint_conversion.to_huggingface.",
  )
  parser.add_argument(
      "--hf-access-token",
      default="",
      help="Optional Hugging Face access token forwarded to the chained export for gated models/tokenizers.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = parse_args(argv)
  to_hf_command = _build_to_huggingface_command(args) if args.hf_model_path else None
  subprocess.run(_build_to_maxtext_command(args), check=True)
  if to_hf_command:
    subprocess.run(to_hf_command, check=True)


if __name__ == "__main__":
  main()

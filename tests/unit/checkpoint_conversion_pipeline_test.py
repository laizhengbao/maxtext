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

import unittest
from unittest import mock

from maxtext.checkpoint_conversion.standalone_scripts import convert_llama_mistral_pipeline


_FAKE_EXPORT_SUPPORTED_MODELS = frozenset(["llama3.1-8b", "llama3.1-70b", "llama3.1-405b"])


class CheckpointConversionPipelineTest(unittest.TestCase):

  @mock.patch("subprocess.run")
  def test_runs_only_maxtext_conversion_without_hf_export(self, mock_run):
    convert_llama_mistral_pipeline.main([
        "--base-model-path=/tmp/meta-ckpt",
        "--maxtext-model-path=gs://bucket/maxtext/llama3.1-8b",
        "--model-size=llama3.1-8b",
    ])

    self.assertEqual(mock_run.call_count, 1)
    command = mock_run.call_args.args[0]
    self.assertIn("maxtext.checkpoint_conversion.standalone_scripts.llama_or_mistral_ckpt", command)
    self.assertIn("--base-model-path", command)
    self.assertIn("/tmp/meta-ckpt", command)
    self.assertIn("--huggingface-checkpoint", command)
    self.assertIn("false", command)

  @mock.patch(
      "maxtext.checkpoint_conversion.standalone_scripts.convert_llama_mistral_pipeline._get_export_supported_models",
      return_value=_FAKE_EXPORT_SUPPORTED_MODELS,
  )
  @mock.patch("subprocess.run")
  def test_runs_hf_export_when_requested(self, mock_run, _mock_supported):
    convert_llama_mistral_pipeline.main([
        "--base-model-path=/tmp/meta-ckpt",
        "--maxtext-model-path=gs://bucket/maxtext/llama3.1-8b",
        "--model-size=llama3.1-8b",
        "--hf-model-path=/tmp/hf-export",
        "--hf-access-token=test-token",
    ])

    self.assertEqual(mock_run.call_count, 2)
    export_command = mock_run.call_args_list[1].args[0]
    self.assertIn("maxtext.checkpoint_conversion.to_huggingface", export_command)
    self.assertIn("model_name=llama3.1-8b", export_command)
    self.assertIn("load_parameters_path=gs://bucket/maxtext/llama3.1-8b/0/items", export_command)
    self.assertIn("base_output_directory=/tmp/hf-export", export_command)
    self.assertIn("hf_access_token=test-token", export_command)

  @mock.patch(
      "maxtext.checkpoint_conversion.standalone_scripts.convert_llama_mistral_pipeline._get_export_supported_models",
      return_value=_FAKE_EXPORT_SUPPORTED_MODELS,
  )
  def test_rejects_unsupported_hf_export_model(self, _mock_supported):
    with self.assertRaisesRegex(ValueError, "--hf-model-path"):
      convert_llama_mistral_pipeline.main([
          "--base-model-path=/tmp/meta-ckpt",
          "--maxtext-model-path=gs://bucket/maxtext/llama2-7b",
          "--model-size=llama2-7b",
          "--hf-model-path=/tmp/hf-export",
      ])


if __name__ == "__main__":
  unittest.main()

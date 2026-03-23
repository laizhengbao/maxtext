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

"""Mllama decoder layers with periodic cross-attention."""

import functools

from flax import nnx
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config, MODEL_MODE_PREFILL
from maxtext.inference import page_manager
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import Dropout, MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, maybe_shard_with_logical


class MllamaDecoderLayer(nnx.Module):
  """Decoder layer for Llama 3.2 Vision / Mllama text backbones."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      use_cross_attention: bool = False,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.use_cross_attention = use_cross_attention

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    dummy_cross_states_shape = (
        batch_size,
        config.max_num_tiles_for_vit * (config.image_size_for_vit // config.patch_size_for_vit) ** 2,
        config.vision_output_dim_for_vit,
    )

    self.input_layernorm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    attention_kwargs = dict(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        prefill_cache_axis_order=tuple(map(int, config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, config.compute_axis_order.split(","))),
        reshape_q=config.reshape_q,
        use_ragged_attention=config.use_ragged_attention,
        ragged_block_size=config.ragged_block_size,
        model_mode=model_mode,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        rngs=rngs,
    )
    if self.use_cross_attention:
      self.cross_attention = Attention(
          inputs_q_shape=dummy_inputs_shape,
          inputs_kv_shape=dummy_cross_states_shape,
          is_nope_layer=True,
          use_qk_norm=True,
          base_kv_cache=False,
          **attention_kwargs,
      )
      self.cross_attn_attn_gate = nnx.Param(jnp.zeros((1,), dtype=config.dtype))
      self.cross_attn_mlp_gate = nnx.Param(jnp.zeros((1,), dtype=config.dtype))
    else:
      self.self_attention = Attention(
          inputs_q_shape=dummy_inputs_shape,
          inputs_kv_shape=dummy_inputs_shape,
          **attention_kwargs,
      )

    self.post_attention_layernorm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )
    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=config.mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        mesh=mesh,
        quant=self.quant,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.dropout = Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=rngs)
    self._maybe_shard_with_logical = functools.partial(
        maybe_shard_with_logical,
        mesh=self.mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
        extra_stack_level=1,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      previous_chunk=None,
      kv_cache=None,
      attention_metadata=None,
      cross_attention_states=None,
  ):
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx_sharding = create_sharding(self.mesh, self.activation_axis_names)
    hidden_states = self.input_layernorm(inputs, out_sharding=lnx_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    if self.use_cross_attention:
      if cross_attention_states is None:
        raise ValueError("Mllama cross-attention layer requires cross_attention_states.")
      attention_output, _ = self.cross_attention(
          hidden_states,
          cross_attention_states,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          slot=slot,
          page_state=page_state,
          previous_chunk=previous_chunk,
          out_sharding=lnx_sharding,
          kv_cache=None,
          attention_metadata=attention_metadata,
      )
      attention_output = self._maybe_shard_with_logical(attention_output, self.activation_axis_names)
      intermediate_inputs = inputs + jnp.tanh(self.cross_attn_attn_gate.value).astype(attention_output.dtype) * attention_output
      kv_cache = None
    else:
      attention_output, kv_cache = self.self_attention(
          hidden_states,
          hidden_states,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          slot=slot,
          page_state=page_state,
          previous_chunk=previous_chunk,
          out_sharding=lnx_sharding,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
      attention_output = self._maybe_shard_with_logical(attention_output, self.activation_axis_names)
      intermediate_inputs = inputs + attention_output

    hidden_states = self.post_attention_layernorm(intermediate_inputs, out_sharding=lnx_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)
    mlp_intermediate_sharding = create_sharding(
        self.mesh, ("activation_batch", "activation_length_no_exp", "activation_mlp")
    )
    mlp_output = self.mlp(
        hidden_states,
        deterministic=deterministic,
        intermediate_sharding=mlp_intermediate_sharding,
        out_sharding=lnx_sharding,
    )
    mlp_output = self._maybe_shard_with_logical(mlp_output, self.activation_axis_names)

    if self.use_cross_attention:
      layer_output = intermediate_inputs + jnp.tanh(self.cross_attn_mlp_gate.value).astype(mlp_output.dtype) * mlp_output
    else:
      layer_output = intermediate_inputs + mlp_output
    layer_output = self.dropout(layer_output, deterministic=deterministic)
    layer_output = self._maybe_shard_with_logical(layer_output, self.activation_axis_names)
    return layer_output, kv_cache


MllamaDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    MllamaDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

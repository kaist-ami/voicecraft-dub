# cp from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/transformer.py, modified by Puyuan Peng 2024
import copy
import numbers
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .activation import MultiheadAttention
from .scaling import ActivationBalancer, BalancedDoubleSwish
from .scaling import BasicNorm as _BasicNorm
import math


_shape_t = Union[int, List[int], torch.Size]


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
            self,
            normalized_shape: _shape_t,
            eps: float = 1e-5,
            elementwise_affine: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            self.bias = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                F.layer_norm(
                    input,
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                ),
                embedding,
            )

        assert embedding is None
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class AdaptiveLayerNorm(nn.Module):
    r"""Adaptive Layer Normalization"""

    def __init__(self, d_model, norm) -> None:
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Tensor = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            weight, bias = torch.split(
                self.project_layer(embedding),
                split_size_or_sections=self.d_model,
                dim=-1,
            )
            return (weight * self.norm(input) + bias, embedding)

        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


class BasicNorm(_BasicNorm):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
    ):
        super(BasicNorm, self).__init__(d_model, eps=eps)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return (
                super(BasicNorm, self).forward(input),
                embedding,
            )

        assert embedding is None
        return super(BasicNorm, self).forward(input)


class BalancedBasicNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
    ):
        super(BalancedBasicNorm, self).__init__()
        self.balancer = ActivationBalancer(
            d_model,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.norm = BasicNorm(d_model, eps, device=device, dtype=dtype)

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            input, embedding = input
            return self.norm((self.balancer(input), embedding))

        assert embedding is None
        return self.norm(self.balancer(input))


class IdentityNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
    ) -> None:
        super(IdentityNorm, self).__init__()

    def forward(self, input: Tensor, embedding: Any = None) -> Tensor:
        if isinstance(input, tuple):
            return input

        assert embedding is None
        return input


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            batch_first: bool = False,
            norm_first: bool = False,
            device=None,
            dtype=None,
            linear1_self_attention_cls: nn.Module = nn.Linear,
            linear2_self_attention_cls: nn.Module = nn.Linear,
            linear1_feedforward_cls: nn.Module = nn.Linear,
            linear2_feedforward_cls: nn.Module = nn.Linear,
            layer_norm_cls: nn.Module = LayerNorm,
            layer_norm_eps: float = 1e-5,
            adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            activation = BalancedDoubleSwish(d_model)

        # # We can't test self.activation in forward() in TorchScript,
        # # so stash some information about it instead.
        # if activation is F.relu or isinstance(activation, torch.nn.ReLU):
        #     self.activation_relu_or_gelu = 1
        # elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
        #     self.activation_relu_or_gelu = 2
        # else:
        #     self.activation_relu_or_gelu = 0
        self.activation = activation

        norm1 = layer_norm_cls(d_model, eps=layer_norm_eps, **factory_kwargs)
        if layer_norm_cls == IdentityNorm:
            norm2 = BalancedBasicNorm(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
        else:
            norm2 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )

        if adaptive_layer_norm:
            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
        else:
            self.norm1 = norm1
            self.norm2 = norm2

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            need_weights: Optional[bool] = False,
            past: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        x, stage_embedding = src, None
        is_src_tuple = False
        if isinstance(src, tuple):
            x, stage_embedding = src
            is_src_tuple = True

        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                    src_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        if need_weights:
            if self.norm_first:
                out, attn = self._sa_block_attn(
                    self.norm1(x, stage_embedding),
                    src_mask,
                    src_key_padding_mask,
                    past
                )
                out, present = out  # present is the kvcache of the present timestep
                x = x + out
                x = x + self._ff_block(self.norm2(x, stage_embedding))
            else:
                out, attn = self._sa_block_attn(x, src_mask, src_key_padding_mask, past)
                out, present = out  # present is the kvcache of the present timestep
                x = self.norm1(
                    x + out,
                    stage_embedding,
                )
                x = self.norm2(x + self._ff_block(x), stage_embedding)
            assert not is_src_tuple
            # return (x, stage_embedding)
            return (x, attn)
        else:
            if self.norm_first:  # HERE!!!
                out = self._sa_block(
                    self.norm1(x, stage_embedding),
                    src_mask,
                    src_key_padding_mask, past
                )
                out, present = out  # present is the kvcache of the present timestep
                x = x + out
                x = x + self._ff_block(self.norm2(x, stage_embedding))
            else:
                out = self._sa_block(x, src_mask, src_key_padding_mask)
                out, present = out  # present is the kvcache of the present timestep
                x = self.norm1(
                    x + out,
                    stage_embedding, past
                )
                x = self.norm2(x + self._ff_block(x), stage_embedding)

            if is_src_tuple:
                x = (x, stage_embedding)
            if present != None:
                x = [x, present]
            return x

    # self-attention block
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            past: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            past=past
        )
        x, present = x
        return self.dropout1(x), present

    # self-attention block, also return attention weights
    def _sa_block_attn(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            past: Optional[Tensor] = None,
    ) -> Tensor:
        x, attn = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            past=past
        )
        x, present = x
        return (self.dropout1(x), present), attn

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


import pdb


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            return_layer_states: bool = False,
            need_weights: Optional[bool] = False,
            past: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """

        if return_layer_states:
            assert not need_weights
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    past=past
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output
        if need_weights:
            assert not return_layer_states
            layer_attn = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    past=past
                )
                layer_attn.append(output[1])

            if self.norm is not None:
                output = self.norm(output)

            return layer_attn, output

        output = src
        all_present = []

        # output: [1,T, 2048] / mask: [1,T,T]
        for n_layer, mod in enumerate(self.layers):
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                past=None if past is None else past[n_layer]
            )
            if isinstance(output, list):
                output, present = output
                all_present.append(present)

        if self.norm is not None:
            output = self.norm(output)
        if all_present != []:
            all_present = torch.stack(all_present, dim=0)  # (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
            output = [output, all_present]
        return output


class TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            linear1_self_attention_cls: nn.Module = nn.Linear,
            linear2_self_attention_cls: nn.Module = nn.Linear,
            linear1_feedforward_cls: nn.Module = nn.Linear,
            linear2_feedforward_cls: nn.Module = nn.Linear,
            batch_first: bool = False,
            norm_first: bool = False,
            device=None,
            dtype=None,
            layer_norm_cls: nn.Module = LayerNorm,
            layer_norm_eps: float = 1e-5,
            adaptive_layer_norm=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            linear1_cls=linear1_self_attention_cls,
            linear2_cls=linear2_self_attention_cls,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = linear1_feedforward_cls(
            d_model, dim_feedforward, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2_feedforward_cls(
            dim_feedforward, d_model, **factory_kwargs
        )

        self.norm_first = norm_first
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        elif isinstance(activation, partial):
            self.activation = activation(d_model)
        elif activation == BalancedDoubleSwish:
            self.activation = BalancedDoubleSwish(d_model)
        else:
            self.activation = activation

        if adaptive_layer_norm:
            norm1 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            norm2 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            norm3 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )

            self.norm1 = AdaptiveLayerNorm(d_model, norm1)
            self.norm2 = AdaptiveLayerNorm(d_model, norm2)
            self.norm3 = AdaptiveLayerNorm(d_model, norm3)
        else:
            self.norm1 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = layer_norm_cls(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )
            if layer_norm_cls == IdentityNorm:
                self.norm3 = BalancedBasicNorm(
                    d_model, eps=layer_norm_eps, **factory_kwargs
                )
            else:
                self.norm3 = layer_norm_cls(
                    d_model, eps=layer_norm_eps, **factory_kwargs
                )

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt_is_tuple = False
        if isinstance(tgt, tuple):
            x, stage_embedding = tgt
            tgt_is_tuple = True
        else:
            x, stage_embedding = tgt, None

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x, stage_embedding), tgt_mask, tgt_key_padding_mask
            )

            x = x + self._mha_block(
                self.norm3(x, stage_embedding),
                memory,
                memory_mask,
                memory_key_padding_mask,
            )

            x = x + self._ff_block(self.norm2(x, stage_embedding))
        else:

            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask),
                stage_embedding,
            )
            x = self.norm3(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask
                ),
                stage_embedding,
            )
            x = self.norm2(x + self._ff_block(x), stage_embedding)

        if tgt_is_tuple:
            return (x, stage_embedding)
        return x

    # self-attention block
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
            self,
            x: Tensor,
            mem: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )


class TransformerEncoder_sb(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_sb, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            memory: Tensor,
            mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            return_layer_states: bool = False,
            need_weights: Optional[bool] = False,
            past: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            return_layer_states: return layers' state (optional).

        Shape:
            see the docs in Transformer class.
        """

        if return_layer_states:  # false
            assert not need_weights
            layer_states = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    past=past
                )
                layer_states.append(output[0])

            if self.norm is not None:
                output = self.norm(output)

            return layer_states, output
        if need_weights:  # false
            assert not return_layer_states
            layer_attn = []  # layers' output
            output = src
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    past=past
                )
                layer_attn.append(output[1])

            if self.norm is not None:
                output = self.norm(output)

            return layer_attn, output

        # output: [1,T, 2048] / mask: [1,T,T]
        output = src
        all_present = []

        for n_layer, mod in enumerate(self.layers):
            output = mod(
                output, memory, tgt_mask=mask, memory_mask=memory_mask, tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)
            # output = mod(
            #     output, memory, src_mask=mask, memory_mask=memory_mask, tgt_key_padding_mask=src_key_padding_mask,
            #     past=None if past is None else past[n_layer]
            # )

            if isinstance(output, list):
                output, present = output
                all_present.append(present)
        if self.norm is not None:
            output = self.norm(output)
        if all_present != []:
            all_present = torch.stack(all_present, dim=0)  # (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
            output = [output, all_present]
        return output


# class SpeechVideoCrossAttention(nn.Module):
#     def __init__(self, embed_dim=2048, num_heads=16, num_video_tokens=10):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_video_tokens = num_video_tokens
#
#         self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#
#         # Learnable positional bias parameter (decay factor)
#         self.alpha = nn.Parameter(torch.tensor(0.1))
#
#         # Layer normalization for stability
#         self.layer_norm = nn.LayerNorm(embed_dim)
#
#     def forward(self, speech_token, video_tokens):
#         batch_size = speech_token.size(0)
#
#         # Positional bias (quadratic decay)
#         positions = torch.arange(self.num_video_tokens, device=speech_token.device).float()
#         positional_bias = -self.alpha * (positions ** 2)  # shape: (10,)
#
#         # Expand positional bias
#         positional_bias = positional_bias.view(1, 1, 1, self.num_video_tokens)  # (1,1,1,10)
#
#         # Compute Q, K, V
#         Q = speech_token  # (B, 1, 2048)
#         K = video_tokens  # (B, 10, 2048)
#         V = video_tokens
#
#         # Compute attention logits manually
#         Q_proj = self.cross_attention.in_proj_q(Q).view(batch_size, 1, self.num_heads, -1).transpose(1, 2)
#         K_proj = self.cross_attention.in_proj_k(K).view(batch_size, self.num_video_tokens, self.num_heads, -1).transpose(1,
#                                                                                                       2)  # (B, heads, 10, d_head)
#         V_proj = self.cross_attention.in_proj_v(V).view(batch_size, self.num_video_tokens, self.num_heads, -1).transpose(1, 2)
#
#         attn_scores = torch.matmul(Q_proj.transpose(1, 2), K_proj.transpose(2, 3)) / math.sqrt(
#             self.embed_dim / self.num_heads)
#         attn_scores += positional_bias  # adding positional bias
#
#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         context = torch.matmul(attn_probs, V_proj.transpose(1, 2))
#
#         # Merge heads
#         context = context.reshape(batch_size, 1, self.embed_dim)
#
#         output = self.cross_attention.out_proj(context)
#
#         # Residual and layer-norm (optional but recommended)
#         output = self.layer_norm(output + speech_token)
#
#         return output, attn_probs

import pdb


class SpeechVideoCrossAttention(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=16, vid_dim=2048, num_video_tokens=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_video_tokens = num_video_tokens
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(vid_dim, embed_dim)
        self.v_proj = nn.Linear(vid_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable positional bias parameter (decay factor)
        self.alpha = nn.Parameter(torch.tensor(0.005))

        # Layer normalization for stability
        # self.layer_norm = nn.LayerNorm(embed_dim,eps=1e-5)
        self.speech_ln = nn.LayerNorm(embed_dim, eps=1e-5)
        self.video_ln = nn.LayerNorm(vid_dim, eps=1e-5)

    def forward(self, speech_token, video_tokens):
        batch_size = speech_token.size(0)

        speech_token = self.speech_ln(speech_token)
        video_tokens = self.video_ln(video_tokens)

        # Positional bias (quadratic decay)
        positions = torch.arange(self.num_video_tokens, device=speech_token.device).float()
        positional_bias = (-self.alpha * (positions ** 2)).clamp(min=-10.0, max=0.0)  # shape: (num_video_tokens,)
        positional_bias = positional_bias.view(1, 1, 1, self.num_video_tokens)  # (1,1,1,num_video_tokens)

        # Project Q, K, V
        Q = self.q_proj(speech_token).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(video_tokens).view(batch_size, self.num_video_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(video_tokens).view(batch_size, self.num_video_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention with positional bias
        attn_scores_ = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, 1, num_video_tokens)
        attn_scores = attn_scores_.clamp(min=-1e3, max=1e3)
        if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
            print("NaNs/Infs detected in attn_scores")

        attn_scores = attn_scores + positional_bias  # adding positional bias
        ####ADDED
        # attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]
        ####ADDED
        attn_probs = torch.softmax(attn_scores, dim=-1)

        if torch.isnan(attn_probs).any() or torch.isinf(attn_probs).any():
            print("NaNs/Infs detected in attn_probs")

        context = torch.matmul(attn_probs, V)  # (B, heads, 1, head_dim)

        # Merge heads
        context = context.transpose(1, 2).reshape(batch_size, 1, self.embed_dim)

        output = self.out_proj(context)

        # Residual connection and layer normalization
        # output = self.layer_norm(output + speech_token)

        if torch.isnan(output).any():
            print("NaNs detected after out_proj")
            print("alpha:", self.alpha.item())
            print("positional_bias:", positional_bias)
            print("attn_scores range:", attn_scores.min().item(), attn_scores.max().item())
            print("attn_probs range:", attn_probs.min().item(), attn_probs.max().item())

        return output, attn_probs

# cp from https://github.com/jasonppy/VoiceCraft modified by Sungbin Kim
import random

import numpy as np
import logging
import argparse, copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .modules.utils import make_pad_mask

from .modules.embedding import SinePositionalEmbedding, TokenEmbedding
from .modules.transformer import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .codebooks_patterns import DelayedPatternProvider

from argparse import Namespace
from huggingface_hub import PyTorchModelHubMixin
import pdb

import sys, logging
import contextlib
from argparse import Namespace

import torch
import torch.nn as nn
from dataclasses import dataclass, field
import sys

sys.path.insert(0, 'fairseq')
from fairseq import checkpoint_utils, utils

def load_avhubert(model_path, modalities, use_cuda=False):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])

    for model in models:
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.eval()
    return models[0], task


def top_k_top_p_filtering(
        logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                            ..., :-1
                                            ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


class VoiceCraft_Dub(
    nn.Module,
    PyTorchModelHubMixin,
):
    def __new__(cls, args: Optional[Namespace] = None, config: Optional[Dict] = None, **kwargs) -> "VoiceCraft_Dub":
        # If initialized from Namespace args => convert to dict config for 'PyTorchModelHubMixin' to serialize it as config.json
        # Won't affect instance initialization
        if args is not None:
            if config is not None:
                raise ValueError("Cannot provide both `args` and `config`.")
            config = vars(args)
        return super().__new__(cls, args=args, config=config, **kwargs)

    def __init__(self, args: Optional[Namespace] = None, config: Optional[Dict] = None):
        super().__init__()

        if args is None:
            if config is None:
                raise ValueError("Either `args` or `config` must be provided.")
            args = Namespace(**config)

        self.args = copy.copy(args)
        self.pattern = DelayedPatternProvider(n_q=self.args.n_codebooks)
        if not getattr(self.args, "special_first", False):
            self.args.special_first = 0
        if not getattr(self.args, "n_special", False):
            self.args.n_special = 3
        self.args.eos = getattr(self.args, "eos", -1)
        self.eog = nn.Parameter(torch.full((self.args.n_codebooks, 1), self.args.eog, dtype=torch.long),requires_grad=False)  # [K 1]
        if self.args.eos > 0:
            assert self.args.eos != self.args.audio_pad_token and self.args.eos != self.args.empty_token, self.args.eos
            self.eos = nn.Parameter(torch.full((self.args.n_codebooks, 1), self.args.eos, dtype=torch.long),requires_grad=False)  # [K 1]
        if isinstance(self.args.audio_vocab_size, str):
            self.args.audio_vocab_size = eval(self.args.audio_vocab_size)

        self.n_text_tokens = self.args.text_vocab_size + 1
        assert self.args.text_pad_token == self.args.text_vocab_size, f"self.args.text_vocab_size: {self.args.text_vocab_size}, self.args.text_pad_token: {self.args.text_pad_token}"

        self.n_audio_tokens = [self.args.audio_vocab_size + self.args.n_special] * self.args.n_codebooks  # special tokens: empty token, EOG token, audio pad token
        assert self.args.audio_vocab_size == self.args.empty_token, self.args.empty_token
        assert self.args.eog == self.args.audio_vocab_size + 1, self.args.eog
        assert self.args.audio_pad_token == self.args.audio_vocab_size + 2, self.args.audio_pad_token

        self.text_embedding = TokenEmbedding(
            dim_model=self.args.d_model,
            vocab_size=self.n_text_tokens,
            dropout=self.args.text_embedding_dropout
        )

        self.audio_embedding = nn.ModuleList(
            [
                TokenEmbedding(
                    dim_model=self.args.audio_embedding_dim,
                    vocab_size=self.n_audio_tokens[k],
                    dropout=self.args.audio_embedding_dropout
                ) for k in range(self.args.n_codebooks)
            ]
        )
        self.mask_embedding = nn.Parameter(torch.randn(self.args.max_n_spans, self.args.d_model), requires_grad=True)
        self.text_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.text_positional_embedding_dropout,
            scale=False,
            alpha=True,  # learnable scaler, scale the volume of positional embedding
        )
        self.audio_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.audio_positional_embedding_dropout,
            scale=False,
            alpha=True,  # learnable scaler, scale the volume of positional embedding
        )

        dec_layer = TransformerEncoderLayer(
            self.args.d_model,
            self.args.nhead,
            dim_feedforward=self.args.d_model * 4,
            dropout=self.args.trm_dropout,
            batch_first=True,
            norm_first=True,
            layer_norm_cls=LayerNorm
        )
        self.decoder = TransformerEncoder(
            dec_layer,
            num_layers=self.args.num_decoder_layers,
            norm=LayerNorm(self.args.d_model),
        )

        self.predict_layer = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.args.d_model, self.args.audio_vocab_size // 2), nn.GELU(),
                              nn.Linear(self.args.audio_vocab_size // 2, self.n_audio_tokens[k])) for k in
                range(self.args.n_codebooks)
            ]
        )

        self.accuracy_metrics = nn.ModuleList(
            [MulticlassAccuracy(
                self.n_audio_tokens[k],
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=None,
            ) for k in range(self.args.n_codebooks)])

        if "avhubert" in sys.modules:
            del sys.modules["avhubert"]
        utils.import_user_module(Namespace(user_dir="./avhubert"))
        avhubert_path = "pretrained_models/avhubert/large_lrs3_iter5.pt"

        modalities = "video, audio"
        use_cuda = torch.cuda.is_available()
        self.avhubert_model, self.avhubert_task = load_avhubert(avhubert_path, modalities, use_cuda=use_cuda)
        if hasattr(self.avhubert_model, 'decoder'):
            print(f"Checkpoint: fine-tuned")
            self.avhubert_model = self.avhubert_model.encoder.w2v_model
        else:
            print(f"Checkpoint: pre-trained w/o fine-tuning")

        for param in self.avhubert_model.parameters():
            param.requires_grad = False
        self.avhubert_model.eval()
        self.fusion_layer = nn.Linear(self.args.d_model * 2, self.args.d_model)

        self.adapting_layer = nn.Sequential(
            nn.Sequential(nn.Linear(1024, self.args.d_model),
                          nn.GELU(),
                          nn.Linear(self.args.d_model, self.args.d_model))
        )

    def shift(self, rearranged_y):
        shifted_y = []
        patterns = []
        for i in range(len(rearranged_y)):
            cur_patterns = [self.pattern.get_pattern(cur_y.shape[1]) for cur_y in rearranged_y[i]]
            out = [cur_pattern.build_pattern_sequence(z=cur_y.unsqueeze(0).contiguous(),special_token=self.args.empty_token, keep_only_valid_steps=False)
                   for cur_pattern, cur_y in zip(cur_patterns, rearranged_y[i])]
            shifted_y.append([item[0].squeeze(0) for item in out])  # the first item is values, later two are indexes and mask
            patterns.append(cur_patterns)
        return shifted_y, patterns

    def insert_mask(self, shifted_y):
        inserted_y = []
        mask_position = []
        mask_value = []

        for i in range(len(shifted_y)):
            num_masks = (len(shifted_y[i]) - 1) // 2
            emb_inds = list(range(self.args.max_n_spans))

            emb_inds_use = emb_inds[:num_masks]
            emb_inds_use = emb_inds_use + emb_inds_use
            mask_value.append(emb_inds_use)
            cur_inserted_y = []
            cur_mask_position = []
            for j in range(len(shifted_y[i]) - 1):
                cur_inserted_y.append(shifted_y[i][j])
                cur_mask_position.append(sum([item.shape[1] for item in cur_inserted_y]))  # each item is of shape [K S], so take shape[1]
                cur_inserted_y.append(self.eog)  # insert mask token of shape [K, 1], BUT we are actually using the eog token as a place holder here, as the real mask will be inserted in embed_y function

            cur_inserted_y.append(shifted_y[i][-1])

            inserted_y.append(cur_inserted_y)
            mask_position.append(cur_mask_position)
        return inserted_y, mask_position, mask_value

    def cat_y(self, inserted_y, mask_position, y_lens):
        reduced_eog = getattr(self.args, "reduced_eog", 0)
        cated_y = []
        new_y_lens = []
        for i in range(len(inserted_y)):
            cur_cated_y = torch.cat(inserted_y[i], dim=1)  # [K S]
            cur_cated_y = cur_cated_y.transpose(1, 0)  # [S K]
            cur_cated_y_len = cur_cated_y.shape[0]
            if reduced_eog:
                assert cur_cated_y_len == y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i]) / 2 + 1), f"cur_cated_y_len == {cur_cated_y_len}, but it should be y_lens[i] ({y_lens[i]}) + len(mask_position[i]) ({len(mask_position[i])}) + (len(mask_position[i]) + 1) * self.args.n_codebooks ({(len(mask_position[i]) + 1) * self.args.n_codebooks}) + (len(mask_position[i])/2 + 1) ({len(mask_position[i]) / 2 + 1})={y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i]) / 2 + 1)}"
            else:
                assert cur_cated_y_len == y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i]) + 1), f"cur_cated_y_len == {cur_cated_y_len}, but it should be y_lens[i] ({y_lens[i]}) + len(mask_position[i]) ({len(mask_position[i])}) + (len(mask_position[i]) + 1) * self.args.n_codebooks ({(len(mask_position[i]) + 1) * self.args.n_codebooks}) + (len(mask_position[i]) + 1) ({len(mask_position[i]) + 1})"  # the last term represent the inserted eog token, originally it's inserted at the end of every token, but this is wrong
            new_y_lens.append(cur_cated_y_len)
            cated_y.append(cur_cated_y)

        cated_y = torch.nn.utils.rnn.pad_sequence(cated_y, batch_first=False, padding_value=self.args.audio_pad_token)
        assert cated_y.shape == torch.Size([max(new_y_lens), len(inserted_y),self.args.n_codebooks]), f"cated_y.shape: {cated_y.shape}, but it should be {torch.Size([max(new_y_lens, len(inserted_y), self.args.n_codebooks)])}"
        cated_y = cated_y.permute(2, 0, 1)  # [T,B,K]->[K,T,B]
        assert cated_y.shape[0] == self.args.n_codebooks, cated_y.shape
        return cated_y, torch.LongTensor(new_y_lens).to(cated_y.device)

    def embed_y(self, cated_y, mask_position, mask_value):
        embedded_y = torch.stack([self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)],
                                 dim=0)  # [K, T, B, D]
        assert embedded_y.shape[0] == self.args.n_codebooks, embedded_y.shape
        assert embedded_y.shape[-1] == self.args.d_model, embedded_y.shape

        embedded_y = embedded_y.sum(dim=0)  # [K,T,B,D]->[T,B,D]
        embedded_y = embedded_y.transpose(1, 0)  # [T,B,D]->[B,T,D]
        for i in range(len(embedded_y)):
            if len(mask_position[i]) > 0:
                embedded_y[i, mask_position[i]] = self.mask_embedding[mask_value[i]]
        return embedded_y

    def prepare_mask(self, y_lens, split_lengths):
        mask_intervals, non_mask_intervals = [], []
        for ii, y_len in enumerate(y_lens):
            split_length = split_lengths[ii]
            non_mask_intervals.append([0, split_length])
            mask_intervals.append([split_length, y_len])

        return mask_intervals, non_mask_intervals

    def prepare_input_target(self, y, y_lens, split_length, vid_input=None, new_split_length=None):
        assert y.shape[1] == self.args.n_codebooks, y.shape
        mask_intervals, non_mask_intervals = self.prepare_mask(y_lens, split_length)

        rearranged_y = []

        for idx, ii in enumerate(split_length):
            rearranged_y.append([y[idx][:, :ii]] + [self.eos] + [torch.cat([y[idx][:, mask_intervals[idx][0]:mask_intervals[idx][1]], self.eog], dim=-1)])
        targets = rearranged_y  # each element in each sample is of shape [K T]

        assert targets[0][0].shape[0] == self.args.n_codebooks, targets[0][0].shape
        shifted_y, patterns = self.shift(rearranged_y)  # each element [K S]
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks, shifted_y[0][0].shape[0]

        inserted_y, mask_position, mask_value = self.insert_mask(shifted_y)

        assert inserted_y[0][0].shape[0] == self.args.n_codebooks, inserted_y[0][0].shape[0]
        assert inserted_y[0][1].shape == torch.Size((self.args.n_codebooks,1)), f"this should be a mask, so should have shape {(self.args.n_codebooks, 1)}, but it's {inserted_y[0][1].shape}"
        # then concat tensors that belong to the same sample (in order) then get the length of each sample, and then stack them in batch dimension, pad them with pad_token
        cated_y, new_y_lens = self.cat_y(inserted_y, mask_position, y_lens)  # KTB
        assert cated_y.shape == torch.Size((self.args.n_codebooks, cated_y.shape[1], len(inserted_y)))

        # embed remember to separately embed the mask tokens
        embedded_y = self.embed_y(cated_y, mask_position, mask_value)  # BTD
        assert embedded_y.shape[1:] == torch.Size((max(new_y_lens), self.args.d_model)), embedded_y.shape

        # positional embedding
        if vid_input == None:
            y_input = self.audio_positional_embedding(embedded_y)

            # make attention mask and padding mask
            y_padding_mask = make_pad_mask(new_y_lens).to(y.device)
            y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y_padding_mask.device)
        else:
            for pp in range(len(embedded_y)):
                vid_len = inserted_y[pp][4].shape[1] - 5
                temp_y = embedded_y[pp:pp + 1, new_split_length[pp]:new_split_length[pp] + vid_len, :]
                embedded_y[pp:pp + 1, new_split_length[pp]:new_split_length[pp] + vid_len,:] = temp_y + self.fusion_layer(torch.concat((temp_y, vid_input[pp:pp + 1, :vid_len, :]), axis=2))

            y_input = self.audio_positional_embedding(embedded_y)
            # make attention mask and padding mask
            y_padding_mask = make_pad_mask(new_y_lens).to(y.device)
            y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y_padding_mask.device)

        return y_input, new_y_lens, targets, y_padding_mask, y_attention_mask, mask_position, patterns

    def remove_mask(self, logits, mask_position, new_y_lens):
        # logits: [B K S card]
        logits_use = []
        for i in range(len(logits)):
            non_mask_positions = [-1] + mask_position[i] + [new_y_lens[i]]
            non_mask_intervals = [[non_mask_positions[i] + 1, non_mask_positions[i + 1]] for i in
                                  range(len(non_mask_positions) - 1)]
            cur_logits_use = [logits[i, :, l:r] for l, r in non_mask_intervals]
            logits_use.append(cur_logits_use)

        return logits_use

    def revert_pattern(self, patterns, logits_use):
        logits_final = []
        logit_masks = []
        for i in range(len(logits_use)):
            cur_logits = [
                item.unsqueeze(0).permute(0, 3, 1, 2).contiguous() for item in logits_use[i]
            ]  # each item is of shape [1 K S card] [1 card K S]
            cur_logits_final = [
                cur_pattern.revert_pattern_logits(
                    item, 0, keep_only_valid_steps=False
                )
                for cur_pattern, item in zip(patterns[i], cur_logits)
            ]  # if input output order doesn't match, this step will give an error
            cur_logits_final_ret = [item[0].permute(0, 2, 3, 1).squeeze(0) for item in cur_logits_final]  # each element is of shape [K,T,card]
            logits_final.append(cur_logits_final_ret)
            logit_masks.append([item[2] for item in cur_logits_final])

        return logits_final, logit_masks

    def dec_forward(
            self,
            x_input,  # [batch, T, 2048)
            x_lens,
            x_attention_mask,  # [T,T]
            x_padding_mask,  # [batch, T]
            y_input,  # [batch, T', 2048]
            new_y_lens,
            y_attention_mask,  # [T',T;]
            y_padding_mask,  # [batch, T']
            past=None,
            last_3_tokens=False
    ):
        x_attn_mask = F.pad(x_attention_mask, (0, new_y_lens.max()), value=True)
        # y_attn_mask: [T', T+T'] => autoregressive manner
        y_attn_mask = F.pad(y_attention_mask, (x_lens.max(), 0), value=False)
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)  # xy_attn_mask: [T+T', T+T']

        # check the valid length of each input
        bsz, src_len = x_input.shape[0], x_lens.max() + new_y_lens.max()
        xy_padding_mask = torch.concat([x_padding_mask, y_padding_mask], dim=1)  # indicate the padding, [batch, T+T']
        _xy_padding_mask = (xy_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.args.nhead, -1, -1).reshape(bsz * self.args.nhead, 1, src_len))  # [batch*16, 1, T+T']

        # Check shapes and resize+broadcast as necessary
        if xy_attn_mask.shape != _xy_padding_mask.shape:
            assert xy_attn_mask.ndim + 1 == _xy_padding_mask.ndim, f"xy_attn_mask.shape: {xy_attn_mask.shape}, _xy_padding_mask: {_xy_padding_mask.shape}"
            xy_attn_mask = xy_attn_mask.unsqueeze(0).repeat(_xy_padding_mask.shape[0], 1, 1)  # Example approach

        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = torch.zeros_like(xy_attn_mask)  # [batch*16, T+T', T+T']
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask

        if past == None:  # do not use kvcache => training
            xy_input = torch.cat([x_input, y_input], dim=1)  # [batch, T+T', 2048]
            out, _ = self.decoder((xy_input, None), mask=xy_attn_mask)  # [1,605,2048], [16,605,605]
            return out[:, (x_lens.max()):], None

        else:  # use kvcache
            xy_input = torch.cat([x_input, vid_input, y_input], dim=1)  # [batch, T+T', 2048]
            if past.ndim > 3:  # uses kvcache, only need to pass the last tokens, this doesn't work with multi-span speech editing yet
                if last_3_tokens:
                    xy_input = xy_input[:, -3:]
                    xy_attn_mask = xy_attn_mask[:, -3:]
                else:
                    xy_input = xy_input[:, -1:]
                    xy_attn_mask = xy_attn_mask[:, -1:]

            out, present = self.decoder((xy_input, None), mask=xy_attn_mask, past=past)
            if isinstance(out, tuple):  # get rid of stage_embedding
                out = out[0]

            if out.shape[1] > x_lens.max():  # the first pass, not kvcache yet
                return out[:, x_lens.max():], present
            else:  # used kvcache
                return out, present

    def forward(self, batch):
        """
        Args:
          x:
            A 2-D tensor of shape (N, S).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (N, K, T).
            where K is the number of codebooks
          vid:
            A 3-D tensor of shape (N, T)
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        """
        x, x_lens, y, y_lens = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"]
        split_length = batch["split_length"]

        if len(x) == 0: return None
        x = x[:,:x_lens.max()]  # this deal with gradient accumulation, where x_lens.max() might not be longer than the length of the current slice of x
        y = y[:, :, :y_lens.max()]
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3 and y.shape[1] == self.args.n_codebooks, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        # makes attention mask and padding mask for x
        x_padding_mask = make_pad_mask(x_lens).to(x.device)
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x_padding_mask.device)

        x_input = self.text_embedding(x)  # convert text tokens into embeddings
        x_input = self.text_positional_embedding(x_input)  # add positional embedding to the input text

        #process video
        vid = batch["vid"]
        vid_lens = batch["vid_lens"]
        vid = vid[:, :vid_lens.max()]

        # makes attention mask and padding mask for x
        vid_padding_mask = make_pad_mask(vid_lens).to(vid.device)

        sample = {"source": {"audio": None, "video": vid.unsqueeze(1).detach(), "padding_mask": vid_padding_mask.detach()}}
        sample = utils.move_to_cuda(sample)

        with torch.no_grad():
            av_hubert_feature, padding_mask = self.avhubert_model.extract_finetune(**sample)  # torch.Size([B, max_T, 1024])
        av_hubert_feature = self.adapting_layer(av_hubert_feature)

        # if self.args.condition != "direct":
        #     ###ADDED FOR TTS TRAINING
        #     y_input, new_y_lens, targets, y_padding_mask, y_attention_mask, mask_position, patterns = self.prepare_input_target(
        #         y, y_lens, split_length)
        #     if self.args.positional == "relative":
        #         av_hubert_feature_ = av_hubert_feature.repeat_interleave(2, dim=1)
        #         new_split_lengths = split_length + 11
        #         vid_input = self.audio_positional_embedding(av_hubert_feature_, new_split_lengths)
        #         ###IF NOT DUPLICATED
        #         # vid_input = vid_input[:, 1::2, :]
        #
        #         ####IF DUPLICATED
        #         vid_lens = vid_lens * 2
        #         vid_padding_mask = make_pad_mask(vid_lens).to(vid.device)
        #         vid_attention_mask = torch.triu(torch.ones(vid.shape[1] * 2, vid.shape[1] * 2), diagonal=1).bool().to(
        #             vid_padding_mask.device)
        #
        #     else:
        #         vid_input = self.video_positional_embedding(av_hubert_feature)
        #
        #     y_out = self.dec_forward_TTS(
        #         x_input,
        #         x_lens,
        #         x_attention_mask,
        #         x_padding_mask,
        #         y_input,
        #         new_y_lens,
        #         y_attention_mask,
        #         y_padding_mask,
        #         vid_input,
        #         vid_lens,
        #         vid_attention_mask,
        #         vid_padding_mask,
        #     )
        # elif self.args.condition == "direct":
        vid_input = av_hubert_feature.repeat_interleave(2, dim=1)
        new_split_lengths = split_length + 11
        # vid_input = self.audio_positional_embedding(av_hubert_feature_, new_split_lengths)

        y_input, new_y_lens, targets, y_padding_mask, y_attention_mask, mask_position, patterns = self.prepare_input_target(y, y_lens, split_length, vid_input=vid_input, new_split_length=new_split_lengths)
        y_out = self.dec_forward(
            x_input,
            x_lens,
            x_attention_mask,
            x_padding_mask,
            y_input,
            new_y_lens,
            y_attention_mask,
            y_padding_mask, )

        ####shared
        # vid_input = self.audio_positional_embedding(av_hubert_feature)
        # vid_lens =vid_lens*2
        # vid_padding_mask = make_pad_mask(vid_lens).to(vid.device)
        # vid_attention_mask = torch.triu(torch.ones(vid.shape[1]*2, vid.shape[1]*2), diagonal=1).bool().to(
        #     vid_padding_mask.device)
        # y_out = self.dec_forward_TTS(
        #     x_input,
        #     x_lens,
        #     x_attention_mask,
        #     x_padding_mask,
        #     y_input,
        #     new_y_lens,
        #     y_attention_mask,
        #     y_padding_mask,
        #     vid_input,
        #     vid_lens,
        #     vid_attention_mask,
        #     vid_padding_mask,
        # )
        ####shared

        ###ADDED FOR TTS TRAINING
        # y_out = self.dec_forward(
        #             x_input,
        #             x_lens,
        #             x_attention_mask,
        #             x_padding_mask,
        #             y_input,
        #             new_y_lens,
        #             y_attention_mask,
        #             y_padding_mask
        #         )

        y_out = y_out[0]  # no kv-caching during training
        assert y_out.shape == y_input.shape, f"y_out.shape: {y_out.shape}, y_input.shape: {y_input.shape}"  # [B S D]

        logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)],dim=1)  # [B K S card]
        # take out the mask token (using mask_position and new_y_lens) and revert (using function provided by self.pattern)
        assert logits.shape[1] == self.args.n_codebooks and logits.shape[3] == self.n_audio_tokens[0], logits.shape

        logits_use = self.remove_mask(logits, mask_position, new_y_lens)

        # revert the pattern shift for each logits section in each sample
        logits_final, logit_masks = self.revert_pattern(patterns, logits_use)
        assert logits_final[0][0].shape[0] == self.args.n_codebooks and logits_final[0][0].shape[2] == \
               self.n_audio_tokens[0], f"it is: {logits_final[0][0].shape}, but should be [K, T, card]"

        # testing
        sample_to_test = 0
        assert len(logits_final[sample_to_test]) == len(
            targets[sample_to_test]), f"{len(logits_final[sample_to_test])}, {len(targets[sample_to_test])}"
        temp = sum([logits_final[sample_to_test][i].shape[:-1] != targets[sample_to_test][i].shape for i in
                    range(len(targets[sample_to_test]))])
        assert temp == 0, f"none equal positions: {temp}, total number of elements: {len(targets[sample_to_test])}"
        logit_masked = sum([(item == False).any() for cur_mask in logit_masks for item in cur_mask])
        assert logit_masked == 0, logit_masks

        logits = torch.cat([torch.cat(item, dim=1) for item in logits_final], dim=1)  # [K, T1+T2+T3+..., card]
        targets = torch.cat([torch.cat(item, dim=1) for item in targets], dim=1)  # [K, T1+T2+T3+...]
        assert targets.shape[0] == logits.shape[0], f"{targets.shape}, {logits.shape}"

        loss = []
        ntokens = []
        top10acc = []

        for k, (logit, target) in enumerate(zip(logits, targets)):
            loss.append(F.cross_entropy(logit, target, reduction='mean'))
            top10acc.append(self.accuracy_metrics[k](logit.detach(), target))
            ntokens.append(len(logit))

        all_ntokens = sum(ntokens)
        if self.args.codebook_weight != None:
            codebook_weight = eval(self.args.codebook_weight)
        else:
            codebook_weight = [1.] * self.args.n_codebooks
        loss = sum([l * nt * cw for l, nt, cw in zip(loss, ntokens, codebook_weight)])
        top10acc_by_codebook = [t10a * nt for t10a, nt in zip(top10acc, ntokens)]
        top10acc = sum(top10acc_by_codebook)
        ntokens = torch.tensor(all_ntokens).to(logits.device)

        return {
            "loss": loss,
            "top10acc": top10acc,
            "top10acc_by_codebook": top10acc_by_codebook,
            "effective_ntoken": ntokens,
        }

    def inference_dubbing_batch(
            self,
            x: torch.Tensor,
            x_lens: torch.Tensor,
            y: torch.Tensor,
            vid,
            top_k: int = -100,
            top_p: float = 1.0,
            temperature: float = 1.0,
            stop_repetition: int = 3,
            kvcache: int = 1,
            silence_tokens: list[int] = [1388, 1898, 131],
            batch_size: int = 10,
            *kargs
    ) -> torch.Tensor:
        """
        different from inference_tts, this implementation uses kvcache, which should have significant speed up
        Args:
          x:
            A 2-D tensor of shape (1, L).
          x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
          y:
            A 3-D tensor of shape (1, T, K).
          top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
          top_p: (`optional`) float
            For Neucleus sampling
          temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        """
        eog_inference = self.args.eos if self.args.eos > 0 else self.args.eog
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2, 1)  # [1,T,K] -> [1,K,T]
        assert y.shape[0] == 1 and y.shape[1] == self.args.n_codebooks, y.shape  # there is no padding

        # make x attention mask and x_input
        x_attention_mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(x.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)

        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)

        ##add video
        vid = vid.unsqueeze(0)
        # makes attention mask and padding mask for x
        sample = {"source": {"audio": None, "video": vid.unsqueeze(1).detach()}}
        sample = utils.move_to_cuda(sample)

        ###
        with torch.no_grad():
            av_hubert_feature, padding_mask = self.avhubert_model.extract_finetune(**sample)  # torch.Size([B, max_T, 1024])
        av_hubert_feature = self.adapting_layer(av_hubert_feature)

        ###mask preparation
        mask_intervals, non_mask_intervals = [], []
        non_mask_intervals.append([0, y_lens[0]])
        mask_intervals.append([y_lens[0], y_lens[0] + av_hubert_feature.shape[1] * 2])
        rearranged_y = []
        rearranged_y.append([y[0][:, :y_lens[0]]] + [self.eos] + [torch.cat([y[0][:, mask_intervals[0][0]:mask_intervals[0][1]], self.eog], dim=-1)])

        shifted_y, patterns = self.shift(rearranged_y)  # each element [K S]
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks, shifted_y[0][0].shape[0]

        inserted_y, mask_position, mask_value = self.insert_mask(shifted_y)

        assert inserted_y[0][0].shape[0] == self.args.n_codebooks, inserted_y[0][0].shape[0]
        assert inserted_y[0][1].shape == torch.Size((self.args.n_codebooks,1)), f"this should be a mask, so should have shape {(self.args.n_codebooks, 1)}, but it's {inserted_y[0][1].shape}"

        # then concat tensors that belong to the same sample (in order) then get the length of each sample, and then stack them in batch dimension, pad them with pad_token
        cated_y, new_y_lens = self.cat_y(inserted_y, mask_position, y_lens)  # KTB
        assert cated_y.shape == torch.Size((self.args.n_codebooks, cated_y.shape[1], len(inserted_y)))

        # embed remember to separately embed the mask tokens
        embedded_y = self.embed_y(cated_y, mask_position, mask_value)  # BTD
        assert embedded_y.shape[1:] == torch.Size((max(new_y_lens), self.args.d_model)), embedded_y.shape
        # drop the last tokens with eog
        embedded_y = embedded_y[:, :-5, :]
        new_y_lens[0] = new_y_lens[0] - 5
        # positional embedding
        y_input = self.audio_positional_embedding(embedded_y)
        # make attention mask and padding mask
        y_padding_mask = make_pad_mask(new_y_lens).to(y.device)
        y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(y_padding_mask.device)
        ###mask perparation
        x_padding_mask = torch.full((1, x_lens[0]), False).to(x.device)
        y_padding_mask = torch.full((1, new_y_lens[0]), False).to(y.device)

        vid_input = av_hubert_feature.repeat_interleave(2, dim=1)
        total_vid_len = vid_input.shape[1]

        # entering the generation stage
        # starting from line 708
        codebook_eog = [False] * self.args.n_codebooks
        generated = []  # doesn't contain any empty token, contain eog
        cur_generated = [[] for _ in range(batch_size)]
        # say 0 is empty, 4 is eog
        # tensor([[ 1,  2,  3,  4,  0,  0],
        #         [ 0,  1,  2,  3,  4,  0],
        #         [ 0,  0,  1,  2,  3,  4]])
        num_gen = []
        cur_num_gen = 0
        ##################### silence repetition handling #####################
        logging.info(f"silence tokens: {silence_tokens}, note that if you are not using the pretrained encodec 6f79c6a8, make sure you specified it yourself, rather than using the default")
        consec_silence_counts = [0 for _ in range(batch_size)]
        prev_tokens = [None for _ in range(batch_size)]
        ##################### silence repetition handling #####################

        # prepare the cache placeholder
        # n_layers, 2, bsz, num_heads, src_len, head_dim
        past = torch.ones([self.args.num_decoder_layers, 2, x.shape[0]], device=x.device,dtype=torch.float32) if kvcache else None
        keep = None

        def sample_helper(n_eog, logits, codebook_eog, top_k, top_p, temperature, prev_tokens, consec_silence_counts,
                          stop_repetition, silence_tokens, cur_num_gen, keep):
            if n_eog == 0:
                logits_adjust = logits
                for jj in range(1, self.args.n_codebooks):
                    logits_adjust[:, jj, eog_inference] = -10000
                    logits_adjust[:, jj, self.args.empty_token] = -10000
                if cur_num_gen <= self.args.encodec_sr // 5:  # this shouldn't happen, but just in case the model stopped too early
                    logits_adjust[:, :, eog_inference] = -10000
                ##################### silence repetition handling #####################
                for b in range(batch_size):
                    prev_token = prev_tokens[b]
                    consec_silence_count = consec_silence_counts[b]
                    if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
                        if logits_adjust[b, 0, prev_token] < 0:
                            logits_adjust[b, 0, prev_token] = logits_adjust[b, 0, prev_token] * (
                                    consec_silence_count - (stop_repetition - 1))
                        else:
                            logits_adjust[b, 0, prev_token] = logits_adjust[b, 0, prev_token] / (
                                    consec_silence_count - (stop_repetition - 1))
                ##################### silence repetition handling #####################
                samples = topk_sampling(
                    logits_adjust.reshape(batch_size * self.args.n_codebooks, logits_adjust.shape[-1]), top_k=top_k,
                    top_p=top_p, temperature=temperature
                )  # [B*K, 1]
                samples = samples.reshape(batch_size, self.args.n_codebooks, 1)
                assert samples.shape == torch.Size(
                    (batch_size, self.args.n_codebooks, 1)), f"samples.shape: {samples.shape}"

                for b in range(batch_size):
                    if cur_num_gen < self.args.n_codebooks - 1:
                        for jj in range(1, self.args.n_codebooks - cur_num_gen):
                            samples[b, -jj, 0] = self.args.empty_token

                    if (
                            samples[b, 0, 0] == eog_inference or torch.argmax(logits[b, 0], dim=-1) == eog_inference or
                            y_input.shape[1] > x_lens[b] * (self.args.encodec_sr // 5)
                    ):  # last one means y is already too long, shouldn't happen, but put it here
                        samples[b, 0, 0] = eog_inference
                        codebook_eog[0] = True
                        keep = b  # NOTE keep is a very important variable, we only return this one, note that if eog shows up in two samples, keep will be overwritten by the later one (or the last one)
                    ##################### silence repetition handling #####################
                    if samples[b, 0, 0] in silence_tokens and samples[b, 0, 0] == prev_tokens[b]:
                        consec_silence_counts[b] += 1
                    else:
                        consec_silence_counts[b] = 0
                    prev_tokens[b] = samples[b, 0, 0]
                ##################### silence repetition handling #####################
                return samples, codebook_eog, prev_tokens, consec_silence_counts, keep
            else:
                assert sum(
                    codebook_eog[i] for i in range(n_eog)) == n_eog, f"codebook_eog: {codebook_eog}, but n_eog: {n_eog}"
                logits_adjust = logits
                for jj in range(n_eog + 1, self.args.n_codebooks):
                    logits_adjust[:, jj, eog_inference] = -10000
                    logits_adjust[:, jj, self.args.empty_token] = -10000
                samples = topk_sampling(
                    logits_adjust.reshape(batch_size * self.args.n_codebooks, logits_adjust.shape[-1]), top_k=top_k,
                    top_p=top_p, temperature=temperature
                )  # [B, K, 1]
                samples = samples.reshape(batch_size, self.args.n_codebooks, 1)
                for jj in range(n_eog):
                    samples[keep, jj, 0] = self.args.empty_token
                samples[keep, n_eog, 0] = eog_inference
                codebook_eog[n_eog] = True
                return samples, codebook_eog, prev_tokens, consec_silence_counts, keep

        total_generate_len = int(av_hubert_feature.shape[1] * 2)

        for gen_idx in range(total_generate_len + 4):
            ###
            if cur_num_gen == 0:
                assert x_input.ndim == 3 and x_input.shape[0] == 1, x_input.shape
                assert x_padding_mask.ndim == 2 and x_padding_mask.shape[0] == 1, x_padding_mask.shape
                assert y_input.ndim == 3 and y_input.shape[0] == 1 and y_input.shape[1] == new_y_lens[0], y_input.shape
                assert embedded_y.ndim == 3 and embedded_y.shape[0] == 1 and embedded_y.shape[1] == new_y_lens[
                    0], embedded_y.shape
                x_input = x_input.repeat(batch_size, 1, 1)
                x_lens = x_lens.repeat(batch_size)
                x_padding_mask = x_padding_mask.repeat(batch_size, 1)
                y_input = y_input.repeat(batch_size, 1, 1)
                new_y_lens = new_y_lens.repeat(batch_size)
                y_padding_mask = y_padding_mask.repeat(batch_size, 1)
                embedded_y = embedded_y.repeat(batch_size, 1,1)  # will be used to concat with newly generated token embedding

                past = past.repeat(1, 1, batch_size) if past != None else None
            else:
                assert x_input.shape[0] == batch_size and x_padding_mask.shape[0] == batch_size and y_input.shape[
                    0] == batch_size and new_y_lens.shape[0] == batch_size, f"x_input.shape: {x_input.shape}, x_padding_mask.shape: {x_padding_mask.shape}, y_input.shape: {y_input.shape}, new_y_lens.shape: {new_y_lens.shape}"

            y_out = self.dec_forward(
                x_input,
                x_lens,
                x_attention_mask,
                x_padding_mask,
                y_input,
                new_y_lens,
                y_attention_mask,
                y_padding_mask, )

            if past != None:
                past = torch.cat([past, present.to(past.dtype)], dim=-2) if past.ndim > 3 else present.to(past.dtype)
            y_out = y_out[0][:, -1:]  # only take the last token

            logits = torch.stack([self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)],dim=1)  # [B K S card], B==S==1, so [1 K 1 card]
            logits = logits.squeeze(2)

            assert logits.shape == torch.Size((batch_size, self.args.n_codebooks, self.n_audio_tokens[0])), f"{logits.shape}"

            n_eog = sum(codebook_eog)
            assert n_eog < self.args.n_codebooks
            if self.args.eos > 0:  # if we are using end-of-sentence token (which is used by default), eog shouldn't be used here, as there is no masked spans
                for jj in range(self.args.n_codebooks):
                    logits[:, jj, self.args.eog] = -10000.
                    if gen_idx < total_generate_len:
                        logits[:, jj, self.args.empty_token] = -10000.
                        logits[:, jj, 2051] = -10000.
                        logits[:, jj, 2050] = -10000.
                    elif gen_idx == total_generate_len:
                        if jj == 0:
                            logits[:, jj, 2051] = 10000.

            try:
                samples, codebook_eog, prev_tokens, consec_silence_counts, keep = sample_helper(n_eog, logits,
                                                                                                codebook_eog, top_k,
                                                                                                top_p, temperature,
                                                                                                prev_tokens,
                                                                                                consec_silence_counts,
                                                                                                stop_repetition,
                                                                                                silence_tokens,
                                                                                                cur_num_gen, keep)
            except:
                pdb.set_trace()
            cur_num_gen += 1
            if sum(codebook_eog) == 0:  # no eog yet, keep batch_size of samples
                assert keep == None
                for b in range(batch_size):
                    cur_generated[b].append(samples[b].squeeze(-1))
            elif sum(codebook_eog) == 1:  # the first eog just showed up in this step
                assert keep != None
                for b in range(batch_size):
                    cur_generated[b].append(samples[b].squeeze(-1))
            else:  # we are generating the rest eogs for the 'keep' sample
                for b in range(batch_size):
                    cur_generated[b].append(samples[b].squeeze(-1))

            samples_emb = torch.stack([self.audio_embedding[k](samples[:, k]) for k in range(self.args.n_codebooks)],dim=1)  # [B, K,1,D]
            assert samples_emb.shape == torch.Size([batch_size, self.args.n_codebooks, 1, self.args.d_model])
            samples_emb = samples_emb.sum(dim=1, keepdim=False)  # [B,1,D]

            if sum(codebook_eog) == self.args.n_codebooks:  # generation for the current span is done
                codebook_eog = [False] * self.args.n_codebooks
                num_gen.append(cur_num_gen)
                cur_num_gen = 0
                generated.append(cur_generated)
                cur_generated = [[] for _ in range(batch_size)]
                break
            else:
                assert samples_emb.shape == torch.Size(
                    (batch_size, 1, self.args.d_model)), f"samples_emb.shape: {samples_emb.shape}"

            embedded_y = torch.cat([embedded_y, samples_emb], dim=1)

            if cur_num_gen < total_vid_len:
                embedded_y[:, -1:, :] = embedded_y[:, -1:, :] + self.fusion_layer(torch.concat((embedded_y[:, -1:, :], vid_input[:, cur_num_gen, :].repeat(batch_size, 1, 1)),axis=2))

            y_input = self.audio_positional_embedding(embedded_y)  # [B T D]
            # make attention mask and padding mask
            y_attention_mask = torch.triu(torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1).bool().to(
                y.device)
            new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device).repeat(batch_size)
            y_padding_mask = torch.full((batch_size, new_y_lens[0]), False).to(y.device)

        assert len(generated) == 1, f"len(generated): {len(generated)}"

        # revert the pattern
        total_gen = []
        for gg in generated[0]:
            flatten_gen = []
            for l, orig_span in enumerate([gg]):
                span = torch.stack(orig_span, dim=0)  # [T, K]
                span = span.transpose(1, 0)  # [K, T]
                assert span.shape[0] == self.args.n_codebooks, span.shape
                unshifted_span = []
                for j, s in enumerate(span):
                    start_from = j
                    end_at = - (self.args.n_codebooks - start_from)
                    unshifted_span.append(s[start_from:end_at])
                unshifted_span = torch.stack(unshifted_span, dim=0)

                assert unshifted_span.shape[1] == num_gen[l] - self.args.n_codebooks, f"len(unshifted_spans[0]): {len(unshifted_span[0])}, num_gen[l]: {num_gen[l]}"

                flatten_gen.append(unshifted_span)
            assert len(flatten_gen) == 1, len(flatten_gen)

            # combine
            res = [y[0], flatten_gen[0]]
            res = torch.cat(res, dim=1).unsqueeze(0)  # [K, new_t] -> [1, K, new_T]

            expected_y_len = y_len + sum([item - self.args.n_codebooks for item in num_gen])
            assert res.shape == torch.Size((1, self.args.n_codebooks,expected_y_len)), f"res.shape: {res.shape}, expected_y_len: {expected_y_len}. y_len + sum([item - self.args.n_codebooks for item in num_gen]): {y_len} + {sum([item - self.args.n_codebooks for item in num_gen])}"

            if self.args.special_first:
                res = res - int(self.args.n_special)
                flatten_gen = flatten_gen - int(self.args.n_special)
            total_gen.append(flatten_gen[0].unsqueeze(0))

        return res, total_gen
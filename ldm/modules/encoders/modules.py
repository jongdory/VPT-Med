import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

import open_clip
from einops import rearrange
from ldm.util import default, count_params
from collections import OrderedDict
from functools import partial
from einops import repeat


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

class PromptGenerator(nn.Module):
    """Class-conditional Prompt."""
    def __init__(self, vocab_size, embedding_size=768, seq_length=1, hidden_size=768, factor_size=16, hidden_dropout_prob=0.1, initializer_range=0.02, prefix='prompt'):
        super(PromptGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.factor_size = factor_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.prefix = prefix
        
        self.cls_embeds = nn.Embedding(vocab_size, embedding_size * factor_size)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-12)
        
        self.dense = nn.Linear(embedding_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.pos_embeds = nn.Parameter(torch.zeros(1, seq_length, embedding_size, factor_size))
        self.factor_embeds = nn.Parameter(torch.zeros(1, 1, 1, factor_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_embeds.weight, std=self.initializer_range)
        nn.init.trunc_normal_(self.dense.weight, std=self.initializer_range)
        nn.init.trunc_normal_(self.pos_embeds, std=self.initializer_range)
        nn.init.trunc_normal_(self.factor_embeds, std=self.initializer_range)

    def forward(self, x, deterministic=True):
        pos_embs = repeat(self.pos_embeds, '1 s d f -> b s d f', b=x.shape[0])
        cls_embs = rearrange(self.cls_embeds(x), 'b 1 (d f) -> b 1 d f', f=self.factor_size)
        
        tokens = repeat(cls_embs, 'b 1 d f -> b s d f', s=self.seq_length)
        tokens += pos_embs

        tokens = torch.sum(tokens * self.factor_embeds, dim=-1)
        
        tokens = self.layer_norm(tokens)
        tokens = self.dense(tokens)
        
        if not deterministic:
            tokens = self.dropout(tokens)

        return tokens

class PromptLearner(nn.Module):
    def __init__(self, tokenizer, embeddings, seq_length=8, ctx_dim=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.seq_length = seq_length

        self.organs = ["Brain MR image", "Abdominal CT"]
        self.modalities = ["CT modality", "T1 modality", "T1ce modality", "T2 modality", "FLAIR modality", "PD modality"]
        self.tasks = ["Translation", "Super-Resolution", "Denoising", "Inpainting"]

        # prompts
        self.organ_prompt = PromptGenerator(vocab_size=len(self.organs), embedding_size=ctx_dim, seq_length=seq_length, hidden_size=ctx_dim)
        self.modality_prompt = PromptGenerator(vocab_size=len(self.modalities), embedding_size=ctx_dim, seq_length=seq_length, hidden_size=ctx_dim)
        self.task_prompt = PromptGenerator(vocab_size=len(self.tasks), embedding_size=ctx_dim, seq_length=seq_length, hidden_size=ctx_dim)

    def forward(self, texts):
        context = []

        organs, mods, tasks,_ = zip(*[text.split(",") for text in texts])

        org_indices = torch.tensor([self.organs.index(reg) for reg in organs]).unsqueeze(1).cuda()
        mod_indices = torch.tensor([self.modalities.index(mod) for mod in mods]).unsqueeze(1).cuda()
        task_indices = torch.tensor([self.tasks.index(task) for task in tasks]).unsqueeze(1).cuda()
        etc_tokens = torch.cat([self.tokenizer(etc) for etc in texts]).cuda()

        org_embs = self.organ_prompt(org_indices)
        mod_embs = self.modality_prompt(mod_indices)
        task_embs = self.task_prompt(task_indices)
        etc_embs = self.embeddings(etc_tokens)[:,:-self.seq_length*3]

        context = torch.cat([org_embs, mod_embs, task_embs, etc_embs], dim=1)

        return context.cuda()


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model
        self.device = device
        self.max_length = max_length
        self.seq_length = 16
        # prompt tuning
        self.prompt_learner = PromptLearner(open_clip.tokenize, self.model.token_embedding, 
                                            seq_length=self.seq_length, ctx_dim=1024)
        
        if freeze: self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name: param.requires_grad = True
            else: param.requires_grad = False

    def forward(self, text):
        z = self.encode_with_transformer(text)
        return z

    def encode_with_transformer(self, text):
        if text[0] == '':
            token = open_clip.tokenize(text).to(self.device)
            x = self.model.token_embedding(token).to(self.device)
        else:
            x = self.prompt_learner(text)
        
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]




from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.Tensor, attention_mask: torch.Tensor, unmasked_value: Union[bool, float]
    ):
        # fmt: off
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `attention_mask` is
        ```
        [[0, 0, 1],
         [1, 1, 1],
         [0, 1, 1]]
        ```
        and `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
        # fmt: on

        # Get the index of the first non-zero value for every sample in the batch.
        # In the above example, indices = [[2], [0], [1]]]
        tmp = torch.arange(attention_mask.shape[1], 0, -1)
        indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)

        # Find the batch indexes that have unattended tokens on the leftmost side (e.g. [0, 0, 1, 1, 1]), for which the first rows of the
        # expanded mask will be completely unattended.
        left_masked_rows = torch.where(indices > 0)[0]

        if left_masked_rows.shape[0] == 0:
            return expanded_mask
        indices = indices[left_masked_rows]

        max_len = torch.max(indices)
        range_tensor = torch.arange(max_len).unsqueeze(0)
        range_tensor = range_tensor.repeat(indices.size(0), 1)

        # Avoid unmasking tokens at relevant target positions (on the row axis), by rather unmasking possibly several times the first row that should always be unmasked as we filtered out the batch above.
        range_tensor[range_tensor >= indices] = 0

        # TODO: we may drop support for 3D attention mask as the refactor from Patrick maybe dropped this case
        if expanded_mask.dim() == 4:
            num_masks = expanded_mask.shape[1]
            if num_masks == 1:
                # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
                mask_slice = (left_masked_rows[:, None], 0, range_tensor)
            else:
                # Broadcast [left_masked_rows, 1, 1], [1, num_masks, 1], [left_masked_rows, 1, max_len]
                mask_slice = (
                    left_masked_rows[:, None, None],
                    torch.arange(num_masks)[None, :, None],
                    range_tensor[:, None, :],
                )
        else:
            # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
            mask_slice = (left_masked_rows[:, None], range_tensor)

        expanded_mask[mask_slice] = unmasked_value

        return expanded_mask
    
def _create_4d_causal_attention_mask(
    input_shape: Union[torch.Size, Tuple, List],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`

    Args:
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        device (`int`):
            The torch device the created mask shall have.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, device=device
    )

    return attention_mask


# class PromptLearner(nn.Module):
#     def __init__(self, tokenizer, transformer, m_seqlen=4, n_numseq=6, ctx_dim=768, max_length=77, 
#                  device="cuda", layer="last", layer_idx=None):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.transformer = transformer

#         self.dtype = transformer.dtype
#         self.m_seqlen = m_seqlen
#         self.n_numseq = n_numseq
#         self.max_length = max_length
#         self.device = device

#         self.layer = layer
#         self.layer_idx = layer_idx
#         if layer == "hidden":
#             assert layer_idx is not None
#             assert 0 <= abs(layer_idx) <= 12

#         # context vectors
#         ctx_vectors = torch.zeros(1, m_seqlen*n_numseq, ctx_dim, dtype=self.dtype)
#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

#         # task tokens
#         self.task = ["Translation", "Super-Resolution", "Denoising", "Inpainting"]
#         self.ttx = nn.ParameterDict()
#         for t in self.task:
#             self.ttx[t] = nn.Parameter(torch.zeros(1, 1, 768, dtype=self.dtype))

#         # modality tokens
#         self.modality = ["CT", "T1", "T1ce", "T2", "FLAIR", "PD"]
#         self.mtx = nn.ParameterDict()
#         for mod in self.modality:
#             self.mtx[mod] = nn.Parameter(torch.zeros(1, 1, 768, dtype=self.dtype)) # to be optimized

#         self.init_ctx()

#         # resolution embeddings
#         div = [1, 3, 4, 5, 6]
#         res = [str(int(240/i)) for i in div] + [str(int(256/i)) for i in div] + [str(int(512/i)) for i in div]
#         self.res_emb = [self.embeddings(self.tokenize([r])) for r in res]
#         self.res_emb = OrderedDict(zip(res, self.res_emb))

#         # Region embeddings
#         self.region = ["tumor", "normal"]
#         self.reg_emb = [self.embeddings(self.tokenize([r])) for r in self.region]
#         self.reg_emb = OrderedDict(zip(self.region, self.reg_emb))

#         # start and end of sentence tokens
#         self.sos = self.embeddings(torch.tensor([[49406]]))
#         self.eos = self.embeddings(torch.tensor([[49407]]))

#         self.freeze_and_tune()

#     def freeze_and_tune(self):
#         self.transformer = self.transformer.eval()
#         for param in self.parameters():
#             param.requires_grad = False
#         for param in self.transformer.parameters():
#             param.requires_grad = False

#         # Fine-tune context vectors
#         self.ctx.requires_grad = True
#         for param in self.ttx.values():
#             param.requires_grad = True
#         for param in self.mtx.values():
#             param.requires_grad = True

#         # Freeze embeddings using pre-trained weights
#         for param in self.res_emb.values():
#             param.requires_grad = False
#         for param in self.reg_emb.values():
#             param.requires_grad = False
#         self.sos.requires_grad = False
#         self.eos.requires_grad = False

#     def init_ctx(self):
#         self.ctx.data.normal_(mean=0.0, std=0.02)
#         for param in self.ttx.values():
#             param.data.normal_(mean=0.0, std=0.02)
#         for param in self.mtx.values():
#             param.data.normal_(mean=0.0, std=0.02)


#     def tokenize(self, words):
#         tokens = self.tokenizer(words, truncation=True, max_length=self.max_length, return_length=True,
#                                 return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
        
#         return tokens[:,1:-1] # remove sos and eos tokens
    
#     def embeddings(self, input_ids):
#         input_shape = input_ids.size()

#         hidden_states = self.transformer.text_model.embeddings(input_ids)

#         causal_attention_mask = _create_4d_causal_attention_mask(
#             input_shape, hidden_states.dtype, device=hidden_states.device
#         )

#         encoder_outputs = self.transformer.text_model.encoder(
#             inputs_embeds=hidden_states,
#             causal_attention_mask=causal_attention_mask,
#             output_hidden_states=self.layer=="hidden"
#         )

#         last_hidden_state = encoder_outputs[0]
#         last_hidden_state = self.transformer.text_model.final_layer_norm(last_hidden_state)

#         pooled_output = last_hidden_state[
#             torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
#             input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
#         ]

#         outputs = BaseModelOutputWithPooling(
#             last_hidden_state=last_hidden_state,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )

#         if self.layer == "last":
#             return outputs.last_hidden_state
#         elif self.layer == "pooled":
#             return outputs.pooler_output[:, None, :]
#         else:
#             return outputs.hidden_states[self.layer_idx]
        
#     # for batch training
#     def forward(self, texts):
#         m = self.m_seqlen
#         batch_size = len(texts)

#         # start and end of sentence tokens
#         sos = self.sos.repeat(batch_size, 1, 1).to(self.device)
#         eos = self.eos.repeat(batch_size, 1, 1).to(self.device)

#         # prepare input (task, source_modality, target_modality, source_resolution, target_resolution, region (optional))
#         tasks, src_mod, tgt_mod, src_res, tgt_res, region = zip(*[text.split() for text in texts])

#         # tokenize and embed
#         # task
#         task_embs = torch.cat([self.ttx[task] for task in tasks], dim=0)
#         # modality
#         src_mod_embs = torch.cat([self.mtx[mod] for mod in src_mod], dim=0)
#         tgt_mod_embs = torch.cat([self.mtx[mod] for mod in tgt_mod], dim=0)
#         # resolution
#         res = []
#         for src, tgt, reg in zip(src_res, tgt_res, region):
#             src_emb = self.res_emb[src].to(self.device)
#             tgt_emb = self.res_emb[tgt].to(self.device)
#             res_emb = torch.cat([
#                 self.ctx[:,m*3:m*4].to(self.device), src_emb, 
#                 self.ctx[:,m*4:m*5].to(self.device), tgt_emb
#             ], dim=1)

#             if reg != "-":
#                 reg_emb = self.reg_emb[reg].to(self.device)
#                 res_emb = torch.cat([
#                     res_emb, 
#                     self.ctx[:,m*5:m*6].to(self.device),
#                     reg_emb], dim=1)

#             if res_emb.shape[1] < 7 + m*3:
#                 padding = torch.cat([self.eos] * (8 + m*3 - res_emb.shape[1]), dim=1).to(self.device)
#                 res_emb = torch.cat([res_emb, padding], dim=1)
#             res.append(res_emb)

#         res = torch.cat(res, dim=0)
        
#         prompts = torch.cat([
#             sos,
#             self.ctx[:,:m].repeat(len(texts), 1, 1), task_embs,
#             self.ctx[:,m:m*2].repeat(len(texts), 1, 1), src_mod_embs,
#             self.ctx[:,m*2:m*3].repeat(len(texts), 1, 1), tgt_mod_embs,
#             res, eos
#         ], dim=1)

#         prompts = torch.stack([
#             torch.cat([cp, eos[0].repeat(self.max_length - cp.shape[0], 1)]) if cp.shape[0] < self.max_length else cp
#             for cp in prompts
#         ])

#         return prompts.to(self.device)
    
# class PromptLearner(nn.Module):
#     def __init__(self, tokenizer, transformer, m_seqlen=4, n_numseq=6, ctx_dim=768, max_length=77, 
#                  device="cuda", layer="last", layer_idx=None):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.transformer = transformer

#         self.dtype = transformer.dtype
#         self.m_seqlen = m_seqlen
#         self.n_numseq = n_numseq
#         self.max_length = max_length
#         self.device = device

#         self.layer = layer
#         self.layer_idx = layer_idx
#         if layer == "hidden":
#             assert layer_idx is not None
#             assert 0 <= abs(layer_idx) <= 12

#         # context vectors
#         ctx_vectors = torch.zeros(1, m_seqlen*n_numseq, ctx_dim, dtype=self.dtype)
#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

#         # task tokens
#         self.task = ["Translation", "Super-Resolution", "Denoising", "Inpainting"]
#         self.ttx_emb = [self.embeddings(self.tokenize([t])) for t in self.task]
#         self.ttx_emb = OrderedDict(zip(self.task, self.ttx_emb))

#         # modality tokens
#         self.modality = ["CT", "T1", "T1ce", "T2", "FLAIR", "PD"]
#         self.mtx_emb = [self.embeddings(self.tokenize(["MRI " + m])) if m != "CT" else self.embeddings(self.tokenize([m])) for m in self.modality]
#         self.mtx_emb = OrderedDict(zip(self.modality, self.mtx_emb))

#         # resolution embeddings
#         div = [1, 3, 4, 5, 6]
#         res = [str(int(240/i)) for i in div] + [str(int(256/i)) for i in div] + [str(int(512/i)) for i in div]
#         self.res_emb = [self.embeddings(self.tokenize([r])) for r in res]
#         self.res_emb = OrderedDict(zip(res, self.res_emb))

#         # Region embeddings
#         self.region = ["tumor", "normal"]
#         self.reg_emb = [self.embeddings(self.tokenize([r])) for r in self.region]
#         self.reg_emb = OrderedDict(zip(self.region, self.reg_emb))

#         # start and end of sentence tokens
#         self.sos = self.embeddings(torch.tensor([[49406]]))
#         self.eos = self.embeddings(torch.tensor([[49407]]))

#         self.freeze_and_tune()

#     def freeze_and_tune(self):
#         self.transformer = self.transformer.eval()
#         for param in self.parameters():
#             param.requires_grad = False
#         for param in self.transformer.parameters():
#             param.requires_grad = False

#         # Fine-tune context vectors
#         self.ctx.requires_grad = True


#     def tokenize(self, words):
#         tokens = self.tokenizer(words, truncation=True, max_length=self.max_length, return_length=True,
#                                 return_overflowing_tokens=False, padding=False, return_tensors="pt")["input_ids"]
        
#         return tokens[:,1:-1] # remove sos and eos tokens
    
#     def embeddings(self, input_ids):
#         input_shape = input_ids.size()

#         hidden_states = self.transformer.text_model.embeddings(input_ids)

#         causal_attention_mask = _create_4d_causal_attention_mask(
#             input_shape, hidden_states.dtype, device=hidden_states.device
#         )

#         encoder_outputs = self.transformer.text_model.encoder(
#             inputs_embeds=hidden_states,
#             causal_attention_mask=causal_attention_mask,
#             output_hidden_states=self.layer=="hidden"
#         )

#         last_hidden_state = encoder_outputs[0]
#         last_hidden_state = self.transformer.text_model.final_layer_norm(last_hidden_state)

#         pooled_output = last_hidden_state[
#             torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
#             input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
#         ]

#         outputs = BaseModelOutputWithPooling(
#             last_hidden_state=last_hidden_state,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#         )

#         if self.layer == "last":
#             return outputs.last_hidden_state
#         elif self.layer == "pooled":
#             return outputs.pooler_output[:, None, :]
#         else:
#             return outputs.hidden_states[self.layer_idx]
        
        
#     def forward(self, texts):
#         m = self.m_seqlen
#         res = []
        
#         # tokenize and embed - batched
#         tasks, src_mod, tgt_mod, src_res, tgt_res, region = zip(*[text.split() for text in texts])
#         for task, src_m, tgt_m, src_r, tgt_r, reg in zip(tasks, src_mod, tgt_mod, src_res, tgt_res, region):
#             # task
#             task_emb = self.ttx_emb[task].cuda()
#             # modality
#             src_m_emb = self.mtx_emb[src_m].cuda()
#             tgt_m_emb = self.mtx_emb[tgt_m].cuda()
#             # resolution
#             src_r_emb = self.res_emb[src_r].cuda()
#             tgt_r_emb = self.res_emb[tgt_r].cuda()
#             res_emb = torch.cat([
#                 self.sos.cuda(),
#                 self.ctx[:,:m], task_emb,
#                 self.ctx[:,m:m*2], src_m_emb,
#                 self.ctx[:,m*2:m*3], tgt_m_emb,
#                 self.ctx[:,m*3:m*4], src_r_emb, 
#                 self.ctx[:,m*4:m*5], tgt_r_emb
#             ], dim=1)
            
#             # region (optional, only for inpainting)
#             if reg != "-":
#                 reg_emb = self.reg_emb[reg].cuda()
#                 res_emb = torch.cat([
#                     res_emb, 
#                     self.ctx[:,m*5:m*6],
#                     reg_emb], dim=1)
            
#             # padding to max_length
#             padding = torch.cat([self.eos.cuda()] * (self.max_length - res_emb.shape[1]), dim=1)
#             res_emb = torch.cat([res_emb, padding], dim=1)
#             res.append(res_emb)

#         res = torch.cat(res, dim=0)

#         return res.to(self.device)
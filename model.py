from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    source_vocab_size: int = 4096
    target_vocab_size: int = 8192
    d_model: int = 640
    n_heads: int = 10
    d_ff: int = 2560
    encoder_layers: int = 8
    decoder_layers: int = 8
    max_source_length: int = 192
    max_target_length: int = 256
    dropout: float = 0.1
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by modern decoder transformers."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return self.weight.to(dtype=x.dtype) * x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_position_embeddings).float()
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions):
        # x: [batch, heads, sequence, head_dim], positions: [sequence]
        if positions.numel() == 0:
            return x
        cos = self.cos.index_select(0, positions).to(dtype=x.dtype)
        sin = self.sin.index_select(0, positions).to(dtype=x.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1
        )
        return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout,
        max_position_embeddings=4096,
        use_rotary=True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.rotary = (
            RotaryEmbedding(self.head_dim, max_position_embeddings)
            if use_rotary
            else None
        )
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        batch_size, length, _ = x.shape
        return x.view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        batch_size, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, length, -1)

    def forward(
        self,
        query,
        key_value=None,
        attention_mask=None,
        is_causal=False,
        cache=None,
        query_positions=None,
        key_positions=None,
    ):
        q = self.split_heads(self.q_proj(query))
        if self.rotary is not None and query_positions is not None:
            q = self.rotary(q, query_positions)

        if key_value is None:
            k = self.split_heads(self.k_proj(query))
            v = self.split_heads(self.v_proj(query))
            if self.rotary is not None and key_positions is not None:
                k = self.rotary(k, key_positions)
            if cache is not None:
                k = torch.cat((cache[0], k), dim=2)
                v = torch.cat((cache[1], v), dim=2)
        elif cache is None:
            k = self.split_heads(self.k_proj(key_value))
            v = self.split_heads(self.v_proj(key_value))
            if self.rotary is not None and key_positions is not None:
                k = self.rotary(k, key_positions)
        else:
            k, v = cache

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        output = self.out_proj(self.merge_heads(output))
        return output, (k, v)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        # Keep the parameter budget close to a two-projection GELU FFN:
        # 3 * d_model * hidden ~= 2 * d_model * d_ff.
        hidden_dim = max(8, round((2 * d_ff) / 3 / 8) * 8)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.attention = Attention(
            config.d_model, config.n_heads, config.dropout,
            config.max_source_length,
        )
        self.feed_forward = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask, positions=None):
        output, _ = self.attention(
            self.norm1(x), attention_mask=attention_mask,
            query_positions=positions, key_positions=positions,
        )
        x = x + self.dropout(output)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.norm3 = RMSNorm(config.d_model, config.norm_eps)
        self.self_attention = Attention(
            config.d_model, config.n_heads, config.dropout,
            config.max_target_length,
        )
        self.cross_attention = Attention(
            config.d_model, config.n_heads, config.dropout,
            max(config.max_source_length, config.max_target_length),
            use_rotary=False,
        )
        self.feed_forward = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x,
        encoder_output,
        source_mask,
        self_cache=None,
        cross_cache=None,
        incremental=False,
        target_positions=None,
    ):
        output, self_cache = self.self_attention(
            self.norm1(x),
            is_causal=not incremental,
            cache=self_cache,
            query_positions=target_positions,
            key_positions=target_positions,
        )
        x = x + self.dropout(output)

        output, cross_cache = self.cross_attention(
            self.norm2(x),
            key_value=encoder_output,
            attention_mask=source_mask,
            cache=cross_cache,
        )
        x = x + self.dropout(output)
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x, (self_cache, cross_cache)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.source_embedding = nn.Embedding(config.source_vocab_size, config.d_model)
        self.target_embedding = nn.Embedding(config.target_vocab_size, config.d_model)
        self.encoder = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.encoder_norm = RMSNorm(config.d_model, config.norm_eps)
        self.decoder_norm = RMSNorm(config.d_model, config.norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.target_vocab_size, bias=False)
        self.apply(self.init_weights)
        self.lm_head.weight = self.target_embedding.weight

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def source_mask(self, attention_mask):
        if attention_mask is None:
            return None
        return attention_mask[:, None, None, :].bool()

    def encode(self, source_ids, source_attention_mask=None):
        length = source_ids.shape[1]
        if length > self.config.max_source_length:
            raise ValueError("source sequence is too long")

        positions = torch.arange(length, device=source_ids.device)
        x = self.source_embedding(source_ids)
        x = self.dropout(x)
        mask = self.source_mask(source_attention_mask)
        for layer in self.encoder:
            x = layer(x, mask, positions)
        return self.encoder_norm(x), mask

    def decode(
        self,
        target_ids,
        encoder_output,
        source_mask=None,
        cache=None,
    ):
        incremental = cache is not None
        offset = 0 if cache is None else cache[0][0][0].shape[2]
        length = target_ids.shape[1]
        if offset + length > self.config.max_target_length:
            raise ValueError("target sequence is too long")

        positions = torch.arange(offset, offset + length, device=target_ids.device)
        x = self.target_embedding(target_ids)
        x = self.dropout(x)

        new_cache = []
        for index, layer in enumerate(self.decoder):
            self_cache = None if cache is None else cache[index][0]
            cross_cache = None if cache is None else cache[index][1]
            x, layer_cache = layer(
                x,
                encoder_output,
                source_mask,
                self_cache,
                cross_cache,
                incremental,
                positions,
            )
            new_cache.append(layer_cache)

        return self.lm_head(self.decoder_norm(x)), new_cache

    def forward(self, source_ids, target_ids, source_attention_mask=None):
        encoder_output, source_mask = self.encode(source_ids, source_attention_mask)
        logits, _ = self.decode(target_ids, encoder_output, source_mask)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        source_ids,
        source_attention_mask,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        max_new_tokens=512,
        num_beams=1,
        length_penalty=1.0,
        return_all_beams=False,
    ):
        if num_beams < 1:
            raise ValueError("num_beams must be positive")
        if length_penalty < 0:
            raise ValueError("length_penalty must be non-negative")

        was_training = self.training
        self.eval()
        encoder_output, source_mask = self.encode(source_ids, source_attention_mask)
        steps = min(max_new_tokens, self.config.max_target_length)
        if num_beams == 1:
            output = self._generate_greedy(
                encoder_output,
                source_mask,
                source_ids.shape[0],
                steps,
                bos_token_id,
                eos_token_id,
                pad_token_id,
            )
        else:
            output = self._generate_beam_search(
                encoder_output,
                source_mask,
                steps,
                bos_token_id,
                eos_token_id,
                pad_token_id,
                num_beams,
                length_penalty,
                return_all_beams,
            )
        self.train(was_training)
        return output

    def _generate_greedy(
        self,
        encoder_output,
        source_mask,
        batch_size,
        steps,
        bos_token_id,
        eos_token_id,
        pad_token_id,
    ):
        generated = torch.full(
            (batch_size, steps),
            pad_token_id,
            dtype=torch.long,
            device=encoder_output.device,
        )
        current_token = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=encoder_output.device,
        )
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=encoder_output.device,
        )
        cache = None

        last_step = 0
        for step in range(steps):
            logits, cache = self.decode(
                current_token,
                encoder_output,
                source_mask,
                cache,
            )
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = torch.where(finished, pad_token_id, next_token)
            generated[:, step] = next_token
            current_token = next_token[:, None]
            finished = finished | next_token.eq(eos_token_id)
            last_step = step + 1
            if finished.all():
                break

        return generated[:, :last_step]

    def _generate_beam_search(
        self,
        encoder_output,
        source_mask,
        steps,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        num_beams,
        length_penalty,
        return_all_beams=False,
    ):
        batch_size, source_length, hidden_size = encoder_output.shape
        device = encoder_output.device
        vocab_size = self.config.target_vocab_size

        encoder_output = (
            encoder_output[:, None]
            .expand(batch_size, num_beams, source_length, hidden_size)
            .reshape(batch_size * num_beams, source_length, hidden_size)
        )
        if source_mask is not None:
            source_mask = (
                source_mask[:, None]
                .expand(batch_size, num_beams, *source_mask.shape[1:])
                .reshape(batch_size * num_beams, *source_mask.shape[1:])
            )

        generated = torch.full(
            (batch_size * num_beams, steps),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        current_token = torch.full(
            (batch_size * num_beams, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )
        beam_scores = torch.full(
            (batch_size, num_beams),
            float("-inf"),
            device=device,
        )
        beam_scores[:, 0] = 0.0
        completed = torch.full(
            (batch_size, steps),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        completed_scores = torch.full(
            (batch_size,),
            float("-inf"),
            device=device,
        )
        completed_counts = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        cache = None
        batch_offsets = torch.arange(batch_size, device=device)[:, None] * num_beams

        last_step = 0
        for step in range(steps):
            logits, cache = self.decode(
                current_token,
                encoder_output,
                source_mask,
                cache,
            )
            log_probs = F.log_softmax(logits[:, -1].float(), dim=-1)
            candidate_scores = log_probs.view(
                batch_size, num_beams, vocab_size
            ) + beam_scores[:, :, None]
            top_scores, top_indices = candidate_scores.view(
                batch_size, num_beams * vocab_size
            ).topk(min(2 * num_beams, num_beams * vocab_size), dim=-1)
            top_tokens = top_indices % vocab_size

            # As in standard beam search, only EOS hypotheses ranked inside the
            # current beam width are eligible for the completed-hypothesis bank.
            ranks = torch.arange(top_scores.shape[1], device=device)[None, :]
            eligible_eos = (
                top_tokens.eq(eos_token_id)
                & ranks.lt(num_beams)
                & torch.isfinite(top_scores)
                & ~done[:, None]
            )
            new_completed_counts = eligible_eos.sum(dim=1)
            completed_counts = completed_counts + new_completed_counts
            eos_scores = top_scores.masked_fill(~eligible_eos, float("-inf"))
            best_eos_scores, best_eos_ranks = eos_scores.max(dim=1)
            best_eos_indices = top_indices.gather(1, best_eos_ranks[:, None]).squeeze(1)
            best_eos_parents = best_eos_indices // vocab_size
            best_eos_flat = best_eos_parents + batch_offsets.squeeze(1)
            best_eos_sequences = generated.index_select(0, best_eos_flat).clone()
            best_eos_sequences[:, step] = eos_token_id
            normalized_eos_scores = best_eos_scores / float(step + 1) ** length_penalty
            improve_completed = normalized_eos_scores > completed_scores
            completed[improve_completed] = best_eos_sequences[improve_completed]
            completed_scores = torch.where(
                improve_completed,
                normalized_eos_scores,
                completed_scores,
            )

            # EOS candidates leave the active beam. The next-best unfinished
            # candidates take their places, so completed programs cannot be lost.
            active_scores = top_scores.masked_fill(
                top_tokens.eq(eos_token_id),
                float("-inf"),
            )
            next_scores, active_ranks = active_scores.topk(num_beams, dim=-1)
            next_indices = top_indices.gather(1, active_ranks)
            next_beams = next_indices // vocab_size
            next_tokens = next_indices % vocab_size
            flat_indices = (next_beams + batch_offsets).reshape(-1)

            generated = generated.index_select(0, flat_indices)
            generated[:, step] = next_tokens.reshape(-1)
            cache = self._reorder_cache(cache, flat_indices)
            beam_scores = next_scores
            current_token = next_tokens.reshape(-1, 1)
            last_step = step + 1
            done = done | completed_counts.ge(num_beams) | ~torch.isfinite(
                beam_scores
            ).any(dim=1)
            if done.all():
                break

        active_scores = beam_scores / float(max(last_step, 1)) ** length_penalty
        best_beams = active_scores.argmax(dim=-1)
        best_indices = best_beams + torch.arange(batch_size, device=device) * num_beams
        all_active = generated.view(batch_size, num_beams, steps)
        best_active = generated.index_select(0, best_indices)
        has_completed = torch.isfinite(completed_scores)
        best_active[has_completed] = completed[has_completed]
        if return_all_beams:
            all_active = all_active[:, :, :last_step]
            all_active[:, 0] = best_active[:, :last_step]
            return all_active
        return best_active[:, :last_step]

    @staticmethod
    def _reorder_cache(cache, indices):
        reordered = []
        for self_cache, cross_cache in cache:
            reordered.append(
                (
                    tuple(tensor.index_select(0, indices) for tensor in self_cache),
                    tuple(tensor.index_select(0, indices) for tensor in cross_cache),
                )
            )
        return reordered


def shift_right(labels, bos_token_id, pad_token_id, ignore_index=-100):
    decoder_input_ids = labels.new_full(labels.shape, pad_token_id)
    decoder_input_ids[:, 0] = bos_token_id
    decoder_input_ids[:, 1:] = labels[:, :-1]
    decoder_input_ids.masked_fill_(decoder_input_ids.eq(ignore_index), pad_token_id)
    return decoder_input_ids

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utilities import *
from tokenizer import *
from main import *

# Part 1
class EncoderLayer(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, d_ff, dropout=0.0):
        super(EncoderLayer, self).__init__()
        head_size = n_embd // num_heads
        self.attention = MultiHeadAttention(num_heads, n_embd, head_size, block_size)
        self.feed_forward = FeedForward(n_embd, d_ff, dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x)
        attn_output, attn_maps = self.attention(x)
        x = x + self.dropout1(attn_output)

        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        return x, attn_maps


class Encoder(nn.Module):
    def __init__(self, input_dim, n_embd, num_layers, num_heads, d_ff, max_seq_len, dropout=0.0):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(input_dim, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, num_heads, max_seq_len, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        attn_maps_list = []
        for layer in self.layers:
            x, attn_maps = layer(x)
            for attn_map in attn_maps:
                attn_maps_list.append(attn_map)

        return x.mean(dim=1), attn_maps_list

# FeedForward Classifier
class FeedForward(nn.Module):
    def __init__(self, n_embd, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Part 2
class DecoderLayer(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, d_ff, dropout=0.0):
        super(DecoderLayer, self).__init__()
        head_size = n_embd // num_heads
        self.self_attention = MultiHeadAttention(num_heads, n_embd, head_size, block_size)
        self.feed_forward = FeedForward(n_embd, d_ff)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x)
        attn_output, attn_maps = self.self_attention(x, True)
        x = x + self.dropout1(attn_output)

        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)

        return x, attn_maps

class Decoder(nn.Module):
    def __init__(self, output_dim, n_embd, num_layers, num_heads, d_ff, max_seq_len, dropout=0.0):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(output_dim, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        self.layers = nn.ModuleList([DecoderLayer(n_embd, num_heads, max_seq_len, d_ff, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(n_embd, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        attn_maps_list = []
        for layer in self.layers:
            x, attn_maps = layer(x)
            for attn_map in attn_maps:
                attn_maps_list.append(attn_map)

        x = self.output_proj(x)
        return x, attn_maps_list

# Helpers referenced from nanoGPT
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        if mask is not None:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out, wei


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out_list = []
        attn_maps_list = []
        for h in self.heads:
            out, attn_maps = h(x, mask)
            out_list.append(out)
            attn_maps_list.append(attn_maps)

        out = torch.cat(out_list, dim=-1)
        out = self.dropout(self.proj(out))

        return out, attn_maps_list
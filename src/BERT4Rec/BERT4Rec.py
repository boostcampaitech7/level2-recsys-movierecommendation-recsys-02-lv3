import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math  

class ScaledDotProductAttention(nn.Module):
    """
    Container module for Scaled Dot-Product Attention.

    Scaled Dot-Product Attention:
    Computes attention scores, applies masking, and calculates the weighted sum of values.
    """
    def __init__(self, hidden_units, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate) # dropout rate

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist



class MultiHeadAttention(nn.Module):
    """
    Container module for Multi-Head Attention.

    Multi-Head Attention:
    Computes scaled dot-product attention across multiple heads and combines the results.
    """

    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units

        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate) # scaled dot product attention module을 사용하여 attention 계산
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)

        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합침
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist


class PositionwiseFeedForward(nn.Module):
    """
    Container module for Position-wise Feed-Forward Network.

    Position-wise Feed-Forward Network:
    Applies two linear transformations with a non-linear activation in between for each position independently.
    """
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()

        # SASRec과의 dimension 차이가 존재
        self.W_1 = nn.Linear(hidden_units, 4 * hidden_units)
        self.W_2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, x):
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x)))) # activation: relu -> gelu
        output = self.layerNorm(self.dropout(output) + residual)
        return output


class BERT4RecBlock(nn.Module):
    """
    Container module for a single block in BERT4Rec.

    BERT4Rec Block:
    Consists of a Multi-Head Attention layer followed by a Position-wise Feed-Forward layer.
    """

    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate) # MultiHeadAttention와
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate) # PositionwiseFeedForward 를 연결

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask) # 1. PositionwiseFeedForward
        output_enc = self.pointwise_feedforward(output_enc) # 2. PositionwiseFeedForward 
        return output_enc, attn_dist



class BERT4Rec(nn.Module):
    """
    Container module for BERT4Rec.

    BERT4Rec:
    A sequential recommendation model based on the BERT architecture that uses self-attention and position embeddings.
    """
    def __init__(self, num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device):
        super(BERT4Rec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        self.item_emb = nn.Embedding(num_item + 2, hidden_units, padding_idx=0) 
        self.pos_emb = nn.Embedding(max_len, hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)

        self.blocks = nn.ModuleList([BERT4RecBlock(num_heads, hidden_units, dropout_rate) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_units, num_item + 1)
        
    def forward(self, log_seqs):
        # seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        # positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        # seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        log_seqs = torch.tensor(log_seqs, dtype=torch.long, device=self.device)
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(log_seqs.size(1), device=self.device).unsqueeze(0).expand_as(log_seqs)
        seqs += self.pos_emb(positions)

        seqs = self.emb_layernorm(self.dropout(seqs))
        # mask = torch.BoolTensor(log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device) # mask for zero pad
        mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1).to(self.device)

        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask)
        out = self.out(seqs)
        return out

        
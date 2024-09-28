import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)

    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.

        Args:
            input: [ batch_size, input_len, model_dim ]


        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from OpenNMT
    """

    def __init__(self, head_count, model_dim, model_dim1, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim1,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim1)

    def forward(self, key, value, query, mask=None, additional_mask=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        global query_projected, key_shaped, value_shaped
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # Project key, value, and query.
        key_projected = self.linear_keys(key)
        value_projected = self.linear_values(value)
        query_projected = self.linear_query(query)

        key_shaped = shape(key_projected)
        value_shaped = shape(value_projected)
        query_shaped = shape(query_projected)

        key_len = key_shaped.size(2)
        query_len = query_shaped.size(2)

        # Apply attention dropout and compute context vectors.
        # Original Encoder/Decoder/Decoder self-attention:
        query_shaped = query_shaped / math.sqrt(dim_per_head)
        scores = torch.matmul(query_shaped, key_shaped.transpose(2, 3))
        top_score = scores.view(batch_size, scores.shape[1],
                                query_len, key_len)[:, 0, :, :].contiguous()
        if mask is not None:
            mol_mask = mask.unsqueeze(1).unsqueeze(-1).expand_as(scores)  # b*length1*length2
            text_mask = additional_mask.unsqueeze(1).unsqueeze(1).expand_as(scores)  # b*length1*length2

            # 应用mask，通常是通过将mask中的0值位置在注意力分数中设置为一个非常小的负数（如-1e9）
            scores = scores.masked_fill(mol_mask == 0, -1e18)
            scores = scores.masked_fill(text_mask == 0,-1e18)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value_shaped)
        context = unshape(context)

        output = self.final_linear(context)

        return output, top_score


class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

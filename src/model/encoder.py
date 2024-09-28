import torch
import torch.nn as nn

from src.model.module import PositionwiseFeedForward, LayerNorm, MultiHeadedAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn, context_attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.context_attn = context_attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, graph_embed, mol_mask=None, text_mask=None):
        # input_norm = self.layer_norm(inputs)
        # context, attn = self.self_attn(input_norm, input_norm, input_norm, mol_mask)
        #
        # query = self.dropout(context).squeeze(1) + inputs

        if graph_embed is not None:
            input_norm = self.layer_norm_2(inputs)
            mid, context_attn = self.context_attn(graph_embed, graph_embed, input_norm, mol_mask, text_mask)

            out = self.dropout(mid) + inputs
        else:
            out = inputs
            context_attn = None

        return self.feed_forward(out), context_attn


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, attn_modules):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers

        self.attn_modules = attn_modules
        self.context_attn_modules = attn_modules

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout, self.attn_modules[i], self.context_attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, out, graph_embed=None, mol_mask=None, text_mask=None):
        '''
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        :return:
        '''

        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, graph_embed, mol_mask, text_mask)

        out = self.layer_norm(out)
        return out, attn

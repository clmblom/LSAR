from .positional_encoding import PositionalEncoding
from .transformer_decoder_return_attn import TransformerDecoderReturnAttn
from .transformer_decoder_return_attn_layer import TransformerDecoderReturnAttnLayer
import torch.nn as nn
import torch
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HTRTransformerDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.5, max_tar_len=None, num_layers=6, norm_first=False, pre_read=False):
        super(HTRTransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_tar_len = max_tar_len
        self.decoder_layer = TransformerDecoderReturnAttnLayer(d_model=hidden_size,
                                                            dim_feedforward=1024,
                                                            nhead=4, dropout=0.5,
                                                            activation='gelu',
                                                            norm_first=norm_first)
        if norm_first:
            self.norm = nn.LayerNorm(hidden_size)
        else:
            self.norm = None
        self.transformer_decoder = TransformerDecoderReturnAttn(self.decoder_layer, num_layers=num_layers,
                                                                norm=self.norm)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout=dropout_p, max_len=max_tar_len)
        self.mask = self.generate_square_subsequent_mask(max_tar_len).to(device)
        self.fc = nn.Linear(hidden_size, output_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, target, memory, memory_key_padding_mask=None, sub_mask=None, no_fc=False):
        if sub_mask is not None:
            tgt_mask = self.mask[:sub_mask, :sub_mask]
        else:
            tgt_mask = self.mask

        # target shape: time, batch
        # memory shape: time, batch, hidden_size
        emb = self.embedding(target)*np.sqrt(self.hidden_size) # time, batch, hidden_size
        emb = self.pe(emb) # time, batch, hidden_size
        out, cross_attn, self_attn = self.transformer_decoder(emb, memory,
                                                              tgt_mask=tgt_mask,
                                                              memory_key_padding_mask=memory_key_padding_mask) # time, batch, hidden_size
        if not no_fc:
            out = self.fc(out) # time, batch, output_size
            return out, cross_attn, self_attn
        else:
            out_fc = self.fc(out) # time, batch, output_size
            return out_fc, out

import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class TransformerDecoderReturnAttnLayer(nn.TransformerDecoderLayer):
    """ Taken straight from Pytorch but with the return of attn2 """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, norm_first):
        super(TransformerDecoderReturnAttnLayer, self).__init__(d_model=d_model, nhead=nhead,
                                                                dim_feedforward=dim_feedforward, dropout=dropout,
                                                                activation=activation)
        self.norm_first = norm_first

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> \
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
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
        if self.norm_first:
            tgt = self.norm1(tgt)
            tgt2, self_attn = self.self_attn(tgt, tgt, tgt,
                                             attn_mask=tgt_mask,
                                             key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)

            tgt = self.norm2(tgt)
            tgt2, cross_attn = self.multihead_attn(tgt, memory, memory,
                                                   attn_mask=memory_mask,
                                                   key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)

            tgt = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            tgt2, self_attn = self.self_attn(tgt, tgt, tgt,
                                             attn_mask=tgt_mask,
                                             key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2, cross_attn = self.multihead_attn(tgt, memory, memory,
                                                   attn_mask=memory_mask,
                                                   key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        return tgt, cross_attn, self_attn

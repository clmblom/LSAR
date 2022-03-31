from torch import Tensor
from typing import Optional, Tuple
import torch.nn as nn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransformerDecoderReturnAttn(nn.TransformerDecoder):
    """ Taken straight from Pytorch but with returning attn"""

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoderReturnAttn, self).__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class."""
        output = tgt

        tar_len, batch, _ = tgt.shape
        src_len = memory.shape[0]
        cross_attns = torch.zeros((self.num_layers, batch, tar_len, src_len), device=device)
        self_attns = torch.zeros((self.num_layers, batch, tar_len, tar_len), device=device)
        for i, mod in enumerate(self.layers):
            output, cross_attn, self_attn = mod(output, memory, tgt_mask=tgt_mask,
                                                memory_mask=memory_mask,
                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                memory_key_padding_mask=memory_key_padding_mask)
            cross_attns[i, :] = cross_attn
            self_attns[i, :] = self_attn
        if self.norm is not None:
            output = self.norm(output)
        return output, cross_attns, self_attns

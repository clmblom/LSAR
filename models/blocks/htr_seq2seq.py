import torch.nn as nn
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class HTRSeq2Seq(nn.Module):
    def __init__(self, visual_extractor, visual_encoder, decoder):
        super(HTRSeq2Seq, self).__init__()
        self.visual_extractor = visual_extractor
        self.visual_encoder = visual_encoder
        self.decoder = decoder
        self.init_model()

    def forward(self, source, target=None, source_pad=None, no_fc=False, start_indx=0, pad_indx=2, end_indx=1):
        # source is an image
        # target comes in the form: <S>the_word
        cnn_features = self.visual_extractor(source)
        enc_features = self.visual_encoder(cnn_features)
        if not self.training:
            time = self.decoder.max_tar_len
            batch = source.shape[0]
            preds = torch.zeros((time+1, batch), device=device, dtype=torch.int).fill_(pad_indx)  # fill with padding
            preds[0, :] = start_indx  # start with <S>
            out = torch.zeros((time+1, batch, self.decoder.output_size), device=device)
            out[:, :, pad_indx] = 1 # fill with padding
            stopped = torch.zeros(batch, dtype=torch.bool, device=device)

            for t in range(1, time+1):
                decoding = torch.logical_not(stopped)
                tmp_out, out_no_fc = self.decoder(preds[:t, decoding], enc_features[:, decoding, :],
                                                  sub_mask=t,
                                                  no_fc=True)  # time, batch, output_size
                preds[t, decoding] = torch.argmax(tmp_out[t-1, :, :], dim=-1).int()
                out[t, decoding] = tmp_out[t-1, :]
                stopped = torch.logical_or(stopped, preds[t, :] == end_indx)
                if torch.sum(stopped) == batch:
                    break
            _, cross_attn, self_attn = self.decoder(preds[:time], enc_features,
                                                    sub_mask=time)
            return out[1:, :, :], preds[1:, :], cross_attn, self_attn
        out, cross_attn, self_attn = self.decoder(target, enc_features,
                                                  no_fc=no_fc)
        return out, 0, cross_attn, self_attn

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
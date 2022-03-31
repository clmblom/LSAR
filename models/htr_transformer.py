from .blocks.resnet34_truncated import ResNet34
from .blocks.htr_encoder import HTREncoder
from .blocks.htr_transformer_decoder import HTRTransformerDecoder
from .blocks.htr_seq2seq import HTRSeq2Seq


def transformer_line(hidden_size,
                     ex_feature_size, ex_feature_height, ex_feature_width,
                     tar_len, output_size, num_dec_layers=6, dropout_p=0.5, norm_first=False):
    # TODO: Change this to a config_file
    print("Hidden_size:", hidden_size)
    print("ex_feature_size:", ex_feature_size)
    print("ex_feature_height:", ex_feature_height)
    print("ex_feature_width:", ex_feature_width)
    print("tar_len:", tar_len)
    print("output_size:", output_size)
    print("Number of decoder layers:", num_dec_layers)
    print("Decoder dropout is", dropout_p)
    print("Norm first is ", norm_first)

    extractor = ResNet34(truncate_blocks=1)
    encoder = HTREncoder(hidden_size=hidden_size,
                         feature_size=ex_feature_size,
                         feature_height=ex_feature_height,
                         feature_width=ex_feature_width)
    decoder = HTRTransformerDecoder(hidden_size=hidden_size,
                                    output_size=output_size,
                                    dropout_p=dropout_p,
                                    max_tar_len=tar_len,
                                    num_layers=num_dec_layers,
                                    norm_first=norm_first)
    seq2seq = HTRSeq2Seq(visual_extractor=extractor,
                         visual_encoder=encoder,
                         decoder=decoder)
    return seq2seq

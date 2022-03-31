from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, Strip
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders
import argparse
import string
import re

def main(args):
    args = args.parse_args()
    init_alpha = list(args.characters)
    print("Character tokens:\t", init_alpha)
    text_tokens = ["<S>", "<E>", "<P>", "<UNK>"]
    print("Text tokens:\t", text_tokens)
    if args.classes:
        class_tokens = re.findall(r'<..?.?>', args.classes)
        print("Class tokens:\t", class_tokens)
    else:
        class_tokens = []

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    special_tokens = text_tokens + class_tokens
    print("special tokens\t", special_tokens)

    trainer = BpeTrainer(special_tokens=special_tokens,
                         vocab_size=len(init_alpha),
                         initial_alphabet=init_alpha,
                         limit_alphabet=len(init_alpha)
                         )
    normalizer = normalizers.Sequence([Lowercase()])

    tokenizer.pre_tokenizer = Split(' ', behavior="isolated")
    tokenizer.normalizer = normalizer
    tokenizer.decoder = decoders.BPEDecoder()
    tokenizer.post_processor = TemplateProcessing(
        single="<S> $0 <E>",
        pair=None,
        special_tokens=[("<S>", 0), ("<E>", 1)],
    )
    tokenizer.train_from_iterator("", trainer, length=1)
    tokenizer_name = "a_tokenizer"

    print(f"Saving to 'saved_tokenizers/{tokenizer_name}.json'")
    tokenizer.save(f'saved_tokenizers/{tokenizer_name}.json')
    str_1 = "Testing with a 123 string!"
    output = tokenizer.encode(str_1)
    print("Test string:\t", str_1)
    print("String list:\t", output.tokens)
    print("Encoded list:\t", output.ids)
    print("Decoded string:\t", tokenizer.decode(output.ids))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Stuff')
    args.add_argument('-c', '--characters', default=string.ascii_lowercase + string.digits + ',. ')
    args.add_argument('-cls', '--classes', default='')
    main(args)
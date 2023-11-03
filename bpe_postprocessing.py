from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# change the post-processor of trained tokenizer

# en_tokenizer
en_tokenizer = ByteLevelBPETokenizer(
    "./transformer/Dataset/en_iwslt_train-vocab.json",
    "./transformer/Dataset/en_iwslt_train-merges.txt"
)

en_tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", en_tokenizer.token_to_id("</s>")),
    ("<s>", en_tokenizer.token_to_id("<s>"))
)

en_tokenizer.enable_truncation(max_length=512)
#en_tokenizer.enable_padding(direction='right', pad_id=1)


# de_tokenizer
de_tokenizer = ByteLevelBPETokenizer(
    "./transformer/Dataset/de_iwslt_train-vocab.json",
    "./transformer/Dataset/de_iwslt_train-merges.txt"
)
de_tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", de_tokenizer.token_to_id("</s>")),
    ("<s>", de_tokenizer.token_to_id("<s>"))
)
de_tokenizer.enable_truncation(max_length=512)


# save as the json files
""" en_tokenizer.save("./transformer/Dataset/en_iwslt_train-tokenizer.json")
de_tokenizer.save("./transformer/Dataset/de_iwslt_train-tokenizer.json") """






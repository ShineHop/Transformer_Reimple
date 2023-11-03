from tokenizers import ByteLevelBPETokenizer

with open('./transformer/Dataset/en_corpus_train.txt', encoding="utf8") as f:
    en_corpus = f.readlines()

with open('./transformer/Dataset/de_corpus_train.txt', encoding="utf8") as f:
    de_corpus = f.readlines()

tokenizer = ByteLevelBPETokenizer()


# tokenizer 생성할 때 필요한 코드
""" tokenizer.train(files='./transformer/Dataset/en_corpus_train.txt', vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer.save_model(".", "en_iwslt_train")

tokenizer.train(files='./transformer/Dataset/de_corpus_train.txt', vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer.save_model(".", "de_iwslt_train") """
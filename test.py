import torch
import torch.nn as nn
from bpe_postprocessing import en_tokenizer, de_tokenizer  
from dataset import iwsltDataset, DataLoader
import model

if __name__ == "__main__":
    en_file = './transformer/Dataset/en_corpus_train.txt'  
    de_file = './transformer/Dataset/de_corpus_train.txt'

    with open(en_file, 'r') as file:
        en_sentences = file.readlines()

    with open(de_file, 'r') as file:
        de_sentences = file.readlines()   

    TOTAL_SENTENCES = 26112 
    en_sentences = en_sentences[:TOTAL_SENTENCES]
    de_sentences = de_sentences[:TOTAL_SENTENCES]
    en_sentences = [sentence.rstrip('\n') for sentence in en_sentences]
    de_sentences = [sentence.rstrip('\n') for sentence in de_sentences]

    train_dataset = iwsltDataset(en_sentences, de_sentences)
    print(train_dataset.__len__())    # 26112
    print(train_dataset.__getitem__(0))

    ## DataLoader       # automatically handle batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn) # batch=64
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pe = model.PositionalEncoding(dim_model=512, max_len=512)
    mha = model.MultiHeadAttention(dim_model=512, num_heads=8)

    # 데이터 순회하기
    for data in train_loader:
        print("Data: ", data)
        print(pe(data['pdd_src']))  # Positional Encoding 체크
        break
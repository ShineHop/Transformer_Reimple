from torch.utils.data import Dataset, DataLoader
import torch
#from tokenizers.implementations import ByteLevelBPETokenizer
from bpe_postprocessing import en_tokenizer, de_tokenizer   


# encoder, decoder inputs 형태에 맞게 <EOS>, <BOS> 토큰 redesign
def tokenize(sentence, no_start_token=True, no_end_token=True):
    if no_start_token:
        sentence.remove(0)  # START_TOKEN_ids = 0
    if no_end_token:
        sentence.remove(2)  # END_TOKEN_ids = 2
    return sentence


class iwsltDataset(Dataset):
    def __init__(self, en_sentences, de_sentences):
        super().__init__()
        self.en_sentences = en_sentences
        self.de_sentences = de_sentences

        self.en_tokenizer = en_tokenizer
        self.de_tokenizer = de_tokenizer
    
    def __len__(self):      # return the number of items in dataset
        return len(self.en_sentences)
    
    def __getitem__(self, idx):
        # src, trg 토큰화
        source_sequence = en_tokenizer.encode(self.en_sentences[idx]).ids
        target_sequence = de_tokenizer.encode(self.de_sentences[idx]).ids
        
        # 2nd tokenize 
        source_sequence = tokenize(source_sequence, no_start_token=True, no_end_token=True)  # no tokens
        target_sequence = tokenize(target_sequence, no_start_token=False, no_end_token=True) # BOS token : remained
        return {"src_input" : source_sequence, "trg_input" : target_sequence}

    
    def collate_fn(self, samples):  # data의 길이가 다른 경우 필요
        src_inputs = [torch.tensor(sample['src_input']) for sample in samples] # device='cuda' 추가함 23.11.09
        trg_inputs = [torch.tensor(sample['trg_input']) for sample in samples]


        pdd_src = torch.nn.utils.rnn.pad_sequence(src_inputs, batch_first=True, padding_value=1)
        pdd_trg = torch.nn.utils.rnn.pad_sequence(trg_inputs, batch_first=True, padding_value=1)
        
        batch = {'pdd_src' : pdd_src,
                 'pdd_trg' : pdd_trg}

        return batch
    

        
""" if __name__ == "__main__":
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
    

    # 데이터 순회하기
    for data in train_loader:
        print("Data: ", data) 
        break """
    
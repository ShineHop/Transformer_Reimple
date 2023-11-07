import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pprint import pprint
# from konlpy.tag import Mecab
from nltk.tokenize import word_tokenize as en_tokenizer
import urllib.request
import csv
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
import time
import copy
from collections import defaultdict
import joblib
import gc
import os

from model import Transformer
from dataset import iwsltDataset

if 'device' not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')


model = Transformer(dim_model=512, num_heads=8, max_len=512 , ffn_hidden=8, num_layers=6, enc_voca_size=52000, dec_voca_size=52000, device=device)
print(model)

# train
en_file = './transformer/Dataset/en_corpus_train.txt'  
de_file = './transformer/Dataset/de_corpus_train.txt'

with open(en_file, 'r') as file:
    en_sentences = file.readlines()

with open(de_file, 'r') as file:
    de_sentences = file.readlines()   

TOTAL_SENTENCES = 206112 
en_sentences = en_sentences[:TOTAL_SENTENCES]
de_sentences = de_sentences[:TOTAL_SENTENCES]
en_sentences = [sentence.rstrip('\n') for sentence in en_sentences]
de_sentences = [sentence.rstrip('\n') for sentence in de_sentences]

# validation
en_valid_file = './transformer/Dataset/en_corpus_valid.txt'  
de_valid_file = './transformer/Dataset/de_corpus_valid.txt'

with open(en_valid_file, 'r') as file:
    en_valid_sentences = file.readlines()

with open(de_valid_file, 'r') as file:
    de_valid_sentences = file.readlines()   

TOTAL_SENTENCES = 888 
en_valid_sentences = en_valid_sentences[:TOTAL_SENTENCES]
de_valid_sentences = de_valid_sentences[:TOTAL_SENTENCES]
en_valid_sentences = [sentence.rstrip('\n') for sentence in en_valid_sentences]
de_valid_sentences = [sentence.rstrip('\n') for sentence in de_valid_sentences]

# test
en_test_file = './transformer/Dataset/en_corpus_test.txt'  
de_test_file = './transformer/Dataset/de_corpus_test.txt'

with open(en_test_file, 'r') as file:
    en_test_sentences = file.readlines()

with open(de_test_file, 'r') as file:
    de_test_sentences = file.readlines()   

TOTAL_SENTENCES = 8079 
en_test_sentences = en_test_sentences[:TOTAL_SENTENCES]
de_test_sentences = de_test_sentences[:TOTAL_SENTENCES]
en_test_sentences = [sentence.rstrip('\n') for sentence in en_test_sentences]
de_test_sentences = [sentence.rstrip('\n') for sentence in de_test_sentences]


# dataset
train_dataset = iwsltDataset(en_sentences, de_sentences)
valid_dataset = iwsltDataset(en_valid_sentences, de_valid_sentences)
test_dataset = iwsltDataset(en_test_sentences, de_test_sentences)
print(train_dataset.__len__())    # 206112

## DataLoader       # automatically handle batching and shuffling
train_dataloader = DataLoader(train_dataset, batch_size=64,shuffle=True, collate_fn=train_dataset.collate_fn) # get_item에서 padding을 했기에 collate_function 없어도 됨
valid_dataloader = DataLoader(valid_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)




def criterion(logits: torch.tensor, targets: torch.tensor):
    return nn.CrossEntropyLoss(ignore_index=1)(logits.view(-1,52000), targets.view(-1))

optimizer = torch.optim.Adam(params = model.parameters())

epoch = 2000
from tqdm import tqdm

model.train()
progress = tqdm(range(epoch))
for i in progress:
    batchloss = 0.0
    
    for data in train_dataloader:
        optimizer.zero_grad()

        result = model(data['pdd_src'].float(), data['pdd_trg'].float())    #.cuda() 삭제
        loss = criterion(result(1,0,2), data['pdd_trg'].float().cuda())
        
        loss.backward()
        optimizer.step()
        batchloss += loss
    progress.set_description("{:0.5f}".format(batchloss.cuda().item() / len(train_dataloader)))

# def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
#     # train 모드로 변경
#     model.train()

#     # for the Mixed Precision
#     # Pytorch 예제 : https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
#     scaler = amp.GradScaler()

#     dataset_size = 0
#     running_loss = 0

#     bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
#     for step, (src, trg_input, trg_output) in bar:
#         src = src
#         trg_input = trg_input
#         trg_output = trg_output

#         batch_size = 64

#         with amp.autocast(enabled=True):
#             output = logits = model          #(enc_src = src, dec_src = trg_input)
#             loss = criterion(logits, trg_output)
            
#         # logits (bs, seq_len, VOCAB_SIZE)
#         # trg_output (bs, seq_len)

#         # loss를 Scale
#         # Scaled Grdients를 계산(call)하기 위해 scaled loss를 backward()
#         scaler.scale(loss).backward()
#         # loss.backward()

#         # scaler.step() first unscales the gradients of the optimizer's assigned params.
#         # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
#         # otherwise, optimizer.step() is skipped.
#         scaler.step(optimizer)
#         # optimizer.step()
        
#         # Updates the scale for next iteration.
#         scaler.update()

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # change learning rate by Scheduler
#         if scheduler is not None:
#             scheduler.step()

#         # loss.item()은 loss를 Python Float으로 반환
#         # loss.item()은 batch data의 average loss이므로, sum of loss를 구하기 위해 batch_size를 곱해준다
#         running_loss += loss.item() * batch_size
#         dataset_size += batch_size

#         epoch_loss = running_loss / dataset_size

#         bar.set_postfix(
#             Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
#         )

#     # Garbage Collector
#     gc.collect()

#     return epoch_loss

# def valid_one_epoch(model, dataloader, device, epoch):
#     model.eval()

#     dataset_size = 0
#     running_loss = 0


#     bar = tqdm(enumerate(dataloader), total=len(dataloader))

#     for step, (src, trg_input, trg_output) in bar:
#         src = src.to(device)
#         trg_input = trg_input.to(device)
#         trg_output = trg_output.to(device)

#         batch_size = src.shape[0]

#         output, logits = model(enc_src = src, dec_src = trg_input)
#         loss = criterion(logits, trg_output)

#         running_loss += loss.item() * batch_size
#         dataset_size += batch_size

#         # 실시간으로 정보를 표시하기 위한 epoch loss
#         val_loss = running_loss / dataset_size

#         bar.set_postfix(
#             Epoch=epoch, Valid_Loss=val_loss, LR=optimizer.param_groups[0]["lr"]
#         )

#     gc.collect()

#     return val_loss


# def run_training(
#     model,
#     optimizer,
#     scheduler,
#     device,
#     num_epochs,
#     metric_prefix="",
#     file_prefix="",
#     early_stopping=True,
#     early_stopping_step=10,
# ):
#     if torch.cuda.is_available():
#         print("[INFO] Using GPU:{}\n".format(torch.cuda.get_device_name()))

#     start = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = np.inf
#     history = defaultdict(list)
#     early_stop_counter = 0

#     # num_epochs만큼, train과 val을 실행한다
#     for epoch in range(1, num_epochs + 1):
#         gc.collect()

#         train_epoch_loss = train_one_epoch(
#             model,
#             optimizer,
#             scheduler,
#             dataloader= train_dataloader,
#             device=device,
#             epoch=epoch,
#         )

#         val_loss = valid_one_epoch(
#             model, valid_dataloader, device=device, epoch=epoch
#         )

#         history[f"{metric_prefix}Train Loss"].append(train_epoch_loss)
#         history[f"{metric_prefix}Valid Loss"].append(val_loss)

#         # Log the metrics
#         wandb.log(
#             {
#                 f"{metric_prefix}Train Loss": train_epoch_loss,
#                 f"{metric_prefix}Valid Loss": val_loss,
#             }
#         )

#         print(f"Valid Loss : {val_loss}")

#         # deep copy the model
#         if val_loss <= best_loss:
#             early_stop_counter = 0

#             print(
#                 f"Validation Loss improved( {best_loss} ---> {val_loss}  )"
#             )

#             # Update Best Loss
#             best_loss = val_loss
            
#             # Update Best Model Weight
#             # run.summary['Best RMSE'] = best_loss
#             best_model_wts = copy.deepcopy(model.state_dict())

#             PATH = "{}epoch{:.0f}_Loss{:.4f}.bin".format(file_prefix, epoch, best_loss)
#             torch.save(model.state_dict(), PATH)
#             torch.save(model.state_dict(), f"{file_prefix}best_{epoch}epoch.bin")
#             # Save a model file from the current directory
#             wandb.save(PATH)

#             print(f"Model Saved")

#         elif early_stopping:
#             early_stop_counter += 1
#             if early_stop_counter > early_stopping_step:
#                 break

#         print()

#     end = time.time()
#     time_elapsed = end - start
#     print(
#         "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
#             time_elapsed // 3600,
#             (time_elapsed % 3600) // 60,
#             (time_elapsed % 3600) % 60,
#         )
#     )
#     print("Best Loss: {:.4f}".format(best_loss))

#     # load best model weights
#     model.load_state_dict(best_model_wts)

#     return model, history
# run_training(
#     model = model,
#     optimizer = optimizer,
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=1e-6),
#     device = device,
#     num_epochs = 20,
#     metric_prefix="",
#     file_prefix="",
#     early_stopping=True,
#     early_stopping_step=10,
# )
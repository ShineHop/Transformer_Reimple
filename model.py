import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super(PositionalEncoding, self).__init__()

        self.token_embedding = nn.Embedding(num_embeddings=52000, embedding_dim=dim_model, padding_idx=1)   # num_embeddings : vocab size
        self.pos_encoding = torch.zeros(max_len, dim_model)

        pos = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # pos = tensor([0, 1, 2, 3, 4, 5]) -> (max_len, 1)의 2차원 tensor : 2차원이면서 두번째 차원은 1인 tensor 
        
        _2i = torch.arange(0, dim_model, step=2).float()

        self.pos_encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/dim_model)))
        self.pos_encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/dim_model)))

        #self.register_buffer("pos_encoding", self.pos_encoding)   # it ensures that these tensors are treated properly 
                                                            # when the model is moved to different devices (CPU/GPU) and during model serialization.

    def forward(self, token_embedding:torch.tensor) -> torch.tensor:    # token_embedding을 torch.tensor로 받고, output 또한 torch.tensor로 내보냄
        token_embedding = self.token_embedding(token_embedding)
        seq_len = token_embedding.size(1)   # torch.Size([64, 63, 512])

        # token_embedding + pos encoding
        return token_embedding + self.pos_encoding[:seq_len, :]


class MultiHeadAttention(nn.Module):    
    def __init__(self, dim_model, num_heads):
        super(MultiHeadAttention, self).__init__()  

        assert dim_model % num_heads == 0       # dim_model=512, num_heads=8

        self.dim_model = dim_model
        self.num_heads = num_heads 

        # query, key, value 크기 : dim_model / num_heads   => split할 때 사용
        self.dim_key = self.dim_value =  self.dim_heads = self.dim_model // self.num_heads  # dim_key = dim_value = dim_heads = 64

        self.w_query = nn.Linear(self.dim_model, self.dim_model)
        self.w_key = nn.Linear(self.dim_model, self.dim_model)
        self.w_value = nn.Linear(self.dim_model, self.dim_model)
        
        self.w_concat = nn.Linear(self.dim_model, self.dim_model)

    def splitHeads(self, tensor):
        # split dim_model into head, dim_tensor
        batch_size, length = tensor.size()
        tensor = tensor.view(batch_size, self.num_heads, length, self.dim_heads)
        
        return tensor
    
    def concatHeads(self, tensor):
        batch_size, head, length = tensor.size()
        dim_model = head * self.dim_heads

        tensor = tensor.view(batch_size, length, dim_model)
        return tensor

    def ScaledDotProductAttention(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask=None):      # mask(optional) : key의 문장 길이
        # tensor) key_T = key.T
        # input 값에서 query, key, value가 tensor check  ':torch.Tensor' 삭제
        # batch_size 들어가면 그 크기도 check
        attention_score = torch.matmul(query, key.T) / math.sqrt(self.dim_key)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -(1e-12))

        attention_weights = nn.Softmax(attention_score)
        attention_value = torch.matmul(attention_weights, value)    # value 벡터와 가중합

        return attention_value       

    def forward(self, query, key, value, mask=None):    # concat, linear 
        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        # split query, key, value into small tensor
        query = self.splitHeads(query)
        key = self.splitHeads(key)
        value = self.splitHeads(value)

        # do scale dot product attention
        output = self.ScaledDotProductAttention(query, key, value, mask=None)

        # cocnat -> linear layer
        output = self.concatHeads(output)
        output = self.w_concat(output)        # dim_model * dim_model

        return output


class PositionwiseFeedForward(nn.Module):       # fully connected
    def __init__(self, dim_model, hidden):      # dim_model=512, hidden=2048
        super(PositionwiseFeedForward, self).__init__() 
        self.w1_linear = nn.Linear(dim_model, hidden)
        self.w2_linear = nn.Linear(hidden, dim_model)

    def forward(self, x):       # follow the FFN(x) = max(0, x*W1 + b1)W2 + b2
        x_FFN = self.w1_linear(x)
        x_FFN = nn.ReLU(x_FFN)
        x_FFN = self.w2_linear(x_FFN)

        return x_FFN


class EncoderLayer(nn.Module):      # 6 layers
    def __init__(self, dim_model, ffn_hidden, num_heads):
        super(EncoderLayer, self).__init__()
        self.dim_model = dim_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads

        self.multihead_attention = MultiHeadAttention(self.dim_model, self.num_heads)
        self.normalization1 = nn.LayerNorm(dim_model)

        self.feedforward = PositionwiseFeedForward(self.dim_model, self.ffn_hidden)
        self.normalization2 = nn.LayerNorm(self.dim_model)
    
    def forward(self, inputs):  #src_mask
        attention = self.multihead_attention(query=inputs, key=inputs, value=inputs, mask=None) #src_mask
        attention = self.normalization1(attention + inputs)

        outputs = self.feedforward(attention)
        outputs = self.normalization2(outputs + attention)

        return outputs

class Encoder(nn.Module):   # num_layers = 6
    def __init__(self, enc_voca_size, max_len, dim_model, ffn_hidden, num_heads, num_layers):
        super(Encoder, self).__init__()

        # embedding : positional encoding에서 한 번에 처리함

        self.pos_encoding = PositionalEncoding(dim_model=dim_model, max_len=max_len)
        # multi layers
        self.multi_layers = nn.ModuleList([EncoderLayer(dim_model=dim_model,
                                            ffn_hidden=ffn_hidden,
                                            num_heads=num_heads) for _ in range(num_layers)])
    
    def forward(self, inputs):    #src_mask
        # embedding + pos_encoding
        pos_inputs = self.pos_encoding(inputs)
        
        # Nx layers
        for layer in self.multi_layers:
            outputs = layer(pos_inputs) # src_mask
        
        return outputs

class DecoderLayer(nn.Module):      # 6 layers
    def __init__(self, dim_model, ffn_hidden, num_heads):
        super(DecoderLayer, self).__init__()
        self.dim_model = dim_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads

        self.masked_self_attention = MultiHeadAttention(self.dim_model, self.num_heads)   # masked 필요
        self.normalization1 = nn.LayerNorm(dim_model)

        self.enc_dec_attention = MultiHeadAttention(self.dim_model, self.num_heads)
        self.normalization2 = nn.LayerNorm(dim_model)

        self.feedforward = PositionwiseFeedForward(self.dim_model, ffn_hidden)
        self.normalization3 = nn.LayerNorm(self.dim_model)
    
    def forward(self, enc_inputs, dec_inputs):      # trg_mask, src_mask
        attention = self.masked_self_attention(query=dec_inputs, key=dec_inputs, value=dec_inputs, mask=None)   # trg_mask 필요
        self_att = self.normalization1(attention + dec_inputs)

        outputs = self.feedforward(self_att)
        outputs = self.normalization3(outputs + self_att)

        if enc_inputs is not None:
            attention = self.enc_dec_attention(query=self_att, key=enc_inputs, value=enc_inputs, mask=None) # src_mask
            enc_dec_att = self.normalization2(attention + self_att)

            outputs = self.feedforward(enc_dec_att)
            outputs = self.normalization3(outputs + enc_dec_att)

        return outputs
    
class Decoder(nn.Module):   # num_layers = 6
    def __init__(self, dec_voca_size, max_len, dim_model, ffn_hidden, num_heads, num_layers):
        super(Decoder, self).__init__()

        # self.embedding = nn.Embedding()
        self.pos_encoding = PositionalEncoding(dim_model=dim_model, max_len=max_len)
        # multi layers
        self.multi_layers = nn.ModuleList([DecoderLayer(dim_model=dim_model,
                                            ffn_hidden=ffn_hidden,
                                            num_heads=num_heads) for _ in range(num_layers)])
        self.linear = nn.Linear(dim_model, dec_voca_size)
    def forward(self, src, trg):  #trg_mask, src_mask
        # embedding + pos_encoding
        pos_inputs = self.pos_encoding(trg)
        
        # Nx layers
        for layer in self.multi_layers:
            outputs = layer(pos_inputs, src)   # trg_mask, src_mask
        
        outputs = nn.Linear(outputs)
        outputs = nn.Softmax(outputs)
        
        return outputs

class Transformer(nn.Module):
    def __init__(self, dim_model, num_heads, max_len, ffn_hidden, num_layers, enc_voca_size, dec_voca_size):
        super(Transformer, self).__init__()
    
        # Encoder
        self.encoder = Encoder(enc_voca_size = enc_voca_size,
                        max_len = max_len, 
                        dim_model = dim_model,
                        ffn_hidden = ffn_hidden, 
                        num_heads = num_heads, 
                        num_layers = num_layers)
        
        # Decoder
        self.decoder = Decoder(dec_voca_size = dec_voca_size,
                        max_len = max_len, 
                        dim_model = dim_model,
                        ffn_hidden = ffn_hidden, 
                        num_heads = num_heads, 
                        num_layers = num_layers)
        
    def forward(self, src, trg):
        enc_src = self.encoder(src)
        output = self.decoder(trg, enc_src)

        return output

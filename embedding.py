from torch import nn
import torch
from dataset import en_vocab, en_preprocess, train_dataset
import math

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, emb_size, dropout=0.1, seq_max_len=500):
        super().__init__()

        # 序列中的每个词转为emb向量，而其他形状保持不变
        self.seq_emb = nn.Embedding(vocab_size,emb_size)

        # 为序列中的每个位置准备位置向量，宽度和词向量相同
        position_idx = torch.arange(0,seq_max_len,dtype=torch.float).unsqueeze(-1) # 5000 - (5000, 1)
        position_emb_fill = position_idx*torch.exp(-torch.arange(0, emb_size, 2)*math.log(10000.0/emb_size)) # 选择偶数位置
        pos_encoding = torch.zeros(seq_max_len, emb_size)
        pos_encoding[:, 0::2] = torch.sin(position_emb_fill)
        pos_encoding[:, 1::2] = torch.cos(position_emb_fill)
        self.register_buffer('pos_encoding', pos_encoding) # 固定参数，不参与训练

        # 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):       # x: (batch_size, seq_len)
        x = self.seq_emb(x)     # x: (batch_size, seq_len, emb_size)
        # pos_encoding (1, seq_max_len, emb_size)
        x = x + self.pos_encoding.unsqueeze(0)[:,:x.size()[1],:] # x: (batch_size, seq_len, emb_size)
        return self.dropout(x)

if __name__ == '__main__':
    emb = EmbeddingWithPosition(len(en_vocab), 128)

    en_tokens, en_ids = en_preprocess(train_dataset[0][0]) # 取de句子转词ID序列
    en_ids_tensor = torch.tensor(en_ids, dtype=torch.long)

    emb_result = emb(en_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    print('en_ids_tensor', en_ids_tensor.size(), 'emb_result', emb_result.size())
    print(emb_result)


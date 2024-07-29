from torch import nn
import torch
from dataset import en_vocab, en_preprocess, train_dataset
from embedding import  EmbeddingWithPosition
import math

class MultiheadAttention(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, head):
        super().__init__()
        self.emb_size = emb_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head = head

        self.w_q = nn.Linear(emb_size, head * q_k_size) # 多头
        self.w_k = nn.Linear(emb_size, head * q_k_size)
        self.w_v = nn.Linear(emb_size, head * v_size)

    def forward(self, x_q, x_k_v, attn_mask):
        # x_q: (batch_size, seq_len, head * q_k_size)
        q = self.w_q(x_q)   # q: (batch_size, seq_len, head * q_k_size)
        k = self.w_k(x_k_v) # k: (batch_size, seq_len, head * q_k_size)

        # 多头形状重塑
        q = q.view(q.size()[0], q.size()[1], self.head, self.q_k_size).transpose(1, 2)
        # q: (batch_size, head, seq_len, q_k_size)
        k = k.view(k.size()[0], k.size()[1], self.head, self.q_k_size).transpose(1, 2).transpose(2, 3)
        # k: (batch_size, head, q_k_size, seq_len)

        # 注意力矩阵
        attn = torch.matmul(q, k)/math.sqrt(self.q_k_size)
        # (batch_size, head, seq_len, seq_len) row是q, col是k

        # 注意力分值处理
        # attn_mask: (batch_size, seq_len, seq_len)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.head, -1, -1)
        # attn_mask: (batch_size,head,seq_len,seq_len)
        attn = attn.masked_fill(attn_mask.bool(),-1e9)
        attn = torch.softmax(attn, dim=-1)
        # scores: (batch_size, head, seq_len, seq_len)

        # 注意力与V相乘
        v = self.w_v(x_k_v) # v: (batch_size, seq_len, head * v_size)
        v = v.view(v.size()[0], v.size()[1], self.head, self.v_size).transpose(1,2)
        z = torch.matmul(attn, v) # z: (batch_size, head, seq_len, v_size)
        z = z.transpose(1, 2) # z: (batch_size, head, seq_len, v_size)
        return z.reshape(z.size()[0], z.size()[1], -1)  # z: (batch, seq_len, head * v_size)

if __name__ == '__main__':
    # 测试第一个batch
    emb = EmbeddingWithPosition(len(en_vocab), 128)
    en_tokens, en_ids = en_preprocess(train_dataset[0][0])
    en_ids_tensor = torch.tensor(en_ids, dtype=torch.long)
    emb_result = emb(en_ids_tensor.unsqueeze(0))
    print('emb_result:', emb_result.size())

    # 多头注意力机制
    multihead = MultiheadAttention(emb_size=128, q_k_size=256, v_size=512, head=8)
    attn_mask = torch.zeros((1,en_ids_tensor.size()[0], en_ids_tensor.size()[0]))

    multihead_result = multihead(x_q=emb_result, x_k_v=emb_result, attn_mask=attn_mask)
    print('multihead_result:', multihead_result.size())
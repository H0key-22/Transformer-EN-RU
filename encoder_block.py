from torch import nn
import torch
from multihead_attn import MultiheadAttention
from embedding import EmbeddingWithPosition
from dataset import en_preprocess, train_dataset, en_vocab

class EncoderBlock(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, f_size, head):
        super().__init__()

        self.multihead_attn = MultiheadAttention(emb_size, q_k_size, v_size, head)  # 多头注意力
        self.z_linear = nn.Linear(head * v_size, emb_size) # 调整多头输出尺寸为emb_size
        self.addnorm1 = nn.LayerNorm(emb_size)  # 对last dim做normalization

        # feed-forward结构
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, emb_size)
        )
        self.addnorm2 = nn.LayerNorm(emb_size) # 对last dim做normalization

    def forward(self, x, attn_mask): # x: (batch_size, seq_len, emb_size)
        z = self.multihead_attn(x, x, attn_mask) # z: (batch_size, seq_len, head * v_size)
        z = self.z_linear(z) # z: (batch_size, seq_len, emb_size)
        output1 = self.addnorm1(z + x)    # z: (batch_size, seq_len, emb_size)

        z = self.feedforward(output1)   # z: (batch_size, seq_len, emb_size)
        return self.addnorm2(z + output1)

if __name__ == '__main__':
    # 取一个batch
    emb =  EmbeddingWithPosition((len(en_vocab)), 128)
    en_tokens, en_ids = en_preprocess(train_dataset[0][0])
    en_ids_tensor = torch.tensor(en_ids, dtype=torch.long)
    emb_result = emb(en_ids_tensor.unsqueeze(0))

    attn_mask=torch.zeros((1, en_ids_tensor.size()[0], en_ids_tensor.size()[0])) # batch中每个样本对应1个注意力矩阵

    # 5个Encoder block堆叠
    encoder_blocks = []
    for i in range(5):
        encoder_blocks.append(EncoderBlock(emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8))

    encoder_outputs = emb_result
    for i in range(5):
        encoder_outputs = encoder_blocks[i](encoder_outputs, attn_mask)
    print('encoder outputs: ', encoder_outputs.size())


from torch import nn
import torch
from encoder_block import EncoderBlock
from embedding import EmbeddingWithPosition
from dataset import en_preprocess, train_dataset, en_vocab, PAD_IDX
from config import DEVICE

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, q_k_size, v_size, f_size, head, nblocks, dropout=0.1, seq_max_len=500):
        super().__init__()
        self.emb = EmbeddingWithPosition(vocab_size, emb_size, dropout, seq_max_len)
        self.encoder_blocks = nn.ModuleList()
        for _ in range(nblocks):
            self.encoder_blocks.append(EncoderBlock(emb_size, q_k_size, v_size, f_size, head))

    def forward(self, x):   # x: (batch_size, seq_len)
        pad_mask = (x==PAD_IDX).unsqueeze(1) # pad_mask: (batch_size, 1, seq_len)
        pad_mask = pad_mask.expand(x.size()[0], x.size()[1], x.size()[1]) # pad_mask: (batch_size, seq_len, seq_len)
        pad_mask = pad_mask.to(DEVICE)

        x = self.emb(x)

        for block in self.encoder_blocks:
            x = block(x, pad_mask)
        return x

if __name__ == '__main__':
    # 取2个de句子转ID序列
    en_tokens1, en_ids1 = en_preprocess(train_dataset[0][0])
    en_tokens2, en_ids2 = en_preprocess(train_dataset[1][0])

    # 组成batch并padding对齐
    if len(en_ids1) < len(en_ids2):
        en_ids1.extend([PAD_IDX] * (len(en_ids2) - len(en_ids1)))
    elif len(en_ids2) < len(en_ids1):
        en_ids2.extend([PAD_IDX] * (len(en_ids1) - len(en_ids2)))

    batch = torch.tensor([en_ids1, en_ids2], dtype = torch.long).to(DEVICE)
    print('batch', batch.size())

    # Encoder编码
    encoder = Encoder(vocab_size=len(en_vocab), emb_size=128, q_k_size=256, v_size=128, f_size=512, head=8, nblocks=3).to(DEVICE)
    z = encoder.forward(batch)
    print('encoder output', z.size())
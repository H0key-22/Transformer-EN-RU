import torch
from dataset import ru_preprocess,BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX,ru_vocab
from config import DEVICE,SEQ_MAX_LEN

# de翻译到en
def translate(transformer,ru_sentence):
    # Ru分词
    ru_tokens,ru_ids=ru_preprocess(ru_sentence)
    if len(ru_tokens)>SEQ_MAX_LEN:
        raise Exception('不支持超过{}的句子'.format(SEQ_MAX_LEN))

    # Encoder阶段
    enc_x_batch=torch.tensor([ru_ids],dtype=torch.long).to(DEVICE)      # 准备encoder输入
    encoder_z=transformer.encode(enc_x_batch)    # encoder编码

    # Decoder阶段
    ru_token_ids=[BOS_IDX] # 翻译结果
    while len(ru_token_ids)<SEQ_MAX_LEN:
        dec_x_batch=torch.tensor([ru_token_ids],dtype=torch.long).to(DEVICE)  # 准备decoder输入
        decoder_z=transformer.decode(dec_x_batch,encoder_z,enc_x_batch)   # decoder解碼
        next_token_probs=decoder_z[0,dec_x_batch.size(-1)-1,:]    # 序列下一个词的概率
        next_token_id=torch.argmax(next_token_probs)    # 下一个词ID
        ru_token_ids.append(next_token_id)

        if next_token_id==EOS_IDX:  # 结束符
            break

    # 生成翻译结果
    ru_token_ids=[id for id in ru_token_ids if id not in [BOS_IDX,EOS_IDX,UNK_IDX,PAD_IDX]] # 忽略特殊字符
    ru_tokens=ru_vocab.lookup_tokens(ru_token_ids)    # 词id序列转token序列
    return ' '.join(ru_tokens)


if __name__=='__main__':
    # 加载模型
    transformer=torch.load('checkpoints/model.pth')
    transformer.eval()
    
    ru=translate(transformer,'Tom was different.')
    print(ru)

    '''
    # 测试数据
    for i in range(100):
        de,en=train_dataset[i]
        en1=translate(transformer,de)
        print('{} -> {} -> {}'.format(de,en,en1))
    '''
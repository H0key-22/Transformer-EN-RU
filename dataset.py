import json
import spacy
from spacy.lang.ru import Russian
from torchtext.vocab import build_vocab_from_iterator

# 从文件中读取数据
file_path = 'processed_data.json'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)[:10000]
train_dataset = converted_data = [(item["english"], item["russian"]) for item in data]


en_tokenizer = spacy.load("en_core_web_sm")
ru_tokenizer = Russian()

# 生成词表
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

en_tokens = [] # 英语token列表
ru_tokens = [] # 俄语token列表

for en, ru in train_dataset:
    doc_ru = ru_tokenizer(ru)
    ru_tokens.append([token.text for token in doc_ru])
    doc_en = en_tokenizer(en)
    en_tokens.append([token.text for token in doc_en])

ru_vocab = build_vocab_from_iterator(ru_tokens, specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM], special_first=True)
ru_vocab.set_default_index(UNK_IDX)
en_vocab = build_vocab_from_iterator(en_tokens, specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM], special_first=True)
en_vocab.set_default_index(UNK_IDX)

# 处理句子特征
def ru_preprocess(ru_sentence):
    tokens = [token.text for token in ru_tokenizer(ru_sentence)]
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = ru_vocab(tokens)
    return tokens, ids

def en_preprocess(en_sentence):
    tokens = [token.text for token in en_tokenizer(en_sentence)]
    tokens = [BOS_SYM] + tokens + [EOS_SYM]
    ids = en_vocab(tokens)
    return tokens, ids

if __name__ == '__main__':
    # 词表大小
    print('ru vocab:', len(ru_vocab))
    print('en vocab:', len(en_vocab))

    # 特征预处理
    en_sentence, ru_sentence = train_dataset[0]
    print(ru_sentence)
    print('ru preprocessed sentence:', ru_preprocess(ru_sentence))
    print('en preprocessed sentence:', en_preprocess(en_sentence))
import json
import random

input_file = 'rus.txt'  # 替换为你的输入文件路径
output_file = 'processed_data.json'  # 替换为你的输出JSON文件路径

data = []
past_sentence=''
with open(input_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        line = line.strip()  # 去除首尾空白符
        if not line:
            continue  # 跳过空行

        # 切分数据和注释部分
        parts = line.split('\t')

        # 提取英语和俄语句子对
        if len(parts) >= 2:
            english_sentence = parts[0].strip()
            russian_sentence = parts[1].strip()
            if english_sentence != past_sentence:
            # 添加到数据列表中
                data.append({
                    'english': english_sentence,
                    'russian': russian_sentence
                })
        past_sentence = english_sentence
# 打乱数据集顺序
random.shuffle(data)
print(len(data))

# 将数据写入JSON文件
with open(output_file, 'w', encoding='utf-8') as f_out:
    json.dump(data, f_out, ensure_ascii=False, indent=4)

print("数据处理完成，并已保存为JSON文件。")



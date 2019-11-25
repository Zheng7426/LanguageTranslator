import json
import os

data_file_train = "translation2019zh_train.json"
data_file_valid = "translation2019zh_valid.json"
out_file_en = "output\\translation2019zh_en.txt"
out_file_zh = "output\\translation2019zh_zh.txt"

if not os.path.exists("output"):
    os.makedirs("output")

with open(out_file_en, 'w', encoding="utf-8") as Of_en, open(out_file_zh, 'w', encoding='utf-8') as Of_zh:
    with open(data_file_train, 'r', encoding="utf-8") as If:
        for line in If:
            Of_en.write(json.loads(line)["english"]+"\n")
            Of_zh.write(json.loads(line)["chinese"]+"\n")
    with open(data_file_valid, 'r', encoding="utf-8") as If:
        for line in If:
            Of_en.write(json.loads(line)["english"]+"\n")
            Of_zh.write(json.loads(line)["chinese"]+"\n")


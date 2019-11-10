#import json

max_item_num = 12000
data_file_zh = "UNv1.0.en-zh.zh"
data_file_en = "UNv1.0.en-zh.en"
out_file = "output\\en-zh-"

with open(data_file_en, 'r', encoding="utf-8") as enf, open(data_file_zh, 'r', encoding="utf-8") as zhf:
    file_index = 0
    line_index = 0
    out_json_file = out_file + str(file_index) + ".json"
    of = open(out_json_file, 'w', encoding="utf-8")
    for x, y in zip(enf, zhf):
        line_index += 1
        if line_index % max_item_num == 1:
            of.close()
            file_index += 1
            out_json_file = out_file + str(file_index) + ".json"
            of = open(out_json_file, 'w', encoding="utf-8")
        x = x.strip()
        y = y.strip()
        of.write("{{\"english\": \"{0}\", \"chinese\": \"{1}\"}}\n".format(x, y))
    of.close()





max_sentence_num = 24000
data_file_en = "data\\cmn_en.txt"
data_file_zh = "data\\cmn_zh.txt"
out_file_fix = "output\\en-zh-cmn-"

FORMAT_JSON = 0
FORMAT_GOGGLE = 1
FORMAT_KERAS = 2


def mergeTwoParallelCorpus(lang1, lang2, oformat):
    with open(lang1, 'r', encoding="utf-8") as enf, open(lang2, 'r', encoding="utf-8") as zhf:
        file_index = 0
        line_index = 0
        out_file = out_file_fix + "{:05d}".format(file_index) + (".json" if (oformat == FORMAT_JSON) else ".dat")
        of = open(out_file, 'w', encoding="utf-8")
        for x, y in zip(enf, zhf):
            line_index += 1
            if line_index % max_sentence_num == 1:
                of.close()
                file_index += 1
                print(file_index)
                out_file = out_file_fix + "{:05d}".format(file_index) + (
                    ".json" if (oformat == FORMAT_JSON) else ".dat")
                of = open(out_file, 'w', encoding="utf-8")
            x = x.strip()
            y = y.strip()
            if (oformat == FORMAT_JSON):
                of.write("{{\"english\": \"{0}\", \"chinese\": \"{1}\"}}\n".format(x, y))
            elif (oformat == FORMAT_GOGGLE):
                of.write("{0} <s> {1}\n".format(x, y))
            elif (oformat == FORMAT_KERAS):
                of.write("{0} \t {1}\n".format(x, y))
        of.close()


if __name__ == '__main__':
    mergeTwoParallelCorpus(data_file_en, data_file_zh, FORMAT_GOGGLE)

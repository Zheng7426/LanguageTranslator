# coding: utf-8
import re
import jieba


def process_text_file(read_file_path, write_file_path):
    jieba.load_userdict('jieba/userdict.txt')

    with open('jieba/chinese_stopwords.txt', encoding="utf-8") as f:
        stop_words = [line.strip() for line in f.readlines()]

    with open(read_file_path, 'r', encoding="utf-8") as r:
        with open(write_file_path, 'a', encoding="utf-8") as w:
            for line in r:
                line = line.strip()
                line = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？：；.、~@#￥%……&*（）]+", "", line)
                words = jieba.lcut(line)
                words = [w for w in words if w not in stop_words]
                line = ' '.join(words)
                w.write(line + '\n')


if __name__ == '__main__':
    process_text_file('en-zh\\translation2019zh_zh.txt', 'data\\chinese.txt')
    process_text_file('en-zh\\UNv1.0.en-zh.zh', 'data\\chinese.txt')

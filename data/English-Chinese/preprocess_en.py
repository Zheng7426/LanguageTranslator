import string
import re
from pickle import dump
from unicodedata import normalize


# load doc into memory
def process_text_file(read_file_path, write_file_path):
    with open(read_file_path, 'r', encoding="utf-8") as r:
        with open(write_file_path, 'a', encoding="utf-8") as w:
            for line in r:
                w.write(clean_lines(line) + '\n')


# clean a list of lines
def clean_lines(line):
    # cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    # normalize unicode characters
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    # tokenize on white space
    line = line.split()
    # convert to lower case
    line = [word.lower() for word in line]
    # remove punctuation from each token
    line = [word.translate(table) for word in line]
    # remove non-printable chars form each token
    line = [re_print.sub('', w) for w in line]
    # remove tokens with numbers in them
    line = [word for word in line if word.isalpha()]
    # store as string
    # cleaned.append(' '.join(line))
    return ' '.join(line)


if __name__ == '__main__':
    # load English dataEnglish
    process_text_file('en-zh\\translation2019zh_en.txt', 'data\\english.txt')
    process_text_file('en-zh\\UNv1.0.en-zh.en', 'data\\english.txt')

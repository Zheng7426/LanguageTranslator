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

def process_nmt_file(read_file_path, write_file_path):
    with open(read_file_path, 'r', encoding="utf-8") as r:
        with open(write_file_path, 'a', encoding="utf-8") as w:
            for line in r:
                input_text, target_text, _ = line.split('\t')
                w.write(clean_lines(input_text) + '\n')


# clean a list of lines
def clean_lines(line):
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    # normalize unicode characters
    line = normalize('NFD', line).encode('ascii', 'ignore')
    line = line.decode('UTF-8')
    line = re.sub(r"i'm", "i am", line)
    line = re.sub(r"he's", "he is", line)
    line = re.sub(r"she's", "she is", line)
    line = re.sub(r"it's", "it is", line)
    line = re.sub(r"that's", "that is", line)
    line = re.sub(r"what's", "that is", line)
    line = re.sub(r"where's", "where is", line)
    line = re.sub(r"how's", "how is", line)
    line = re.sub(r"\'ll", " will", line)
    line = re.sub(r"\'ve", " have", line)
    line = re.sub(r"\'re", " are", line)
    line = re.sub(r"\'d", " would", line)
    line = re.sub(r"\'re", " are", line)
    line = re.sub(r"won't", "will not", line)
    line = re.sub(r"can't", "cannot", line)
    line = re.sub(r"n't", " not", line)
    line = re.sub(r"n'", "ng", line)
    line = re.sub(r"'bout", "about", line)
    line = re.sub(r"'til", "until", line)
    line = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", line)
    # tokenize on white space
    line = line.split()
    print(line)
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
    #process_text_file('en-zh\\translation2019zh_en.txt', 'data\\english.txt')
    #process_text_file('en-zh\\UNv1.0.en-zh.en', 'data\\english.txt')
    process_nmt_file('cmn.txt', 'data\\cmn_en.txt')

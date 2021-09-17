
import pickle
import torch
from tqdm import tqdm
import re
from util_common.nlp.word_dictionary import word2index
from util_common.nlp.word_token import word_token
from util_common.nlp.os import read_folder_content, read_content
import random


def main():
    with open('./data/src_all.pkl', 'rb') as f:
        src_all = pickle.load(f)
    with open('./data/tar_all.pkl', 'rb') as f:
        tar_all = pickle.load(f)
    language = 'chinese'
    configure = eval(read_content('configure'))
    src_words = []
    tar_words = []
    src_new = []
    tar_new = []

    for src in tqdm(src_all):
        src = re.sub('\*\*', '', src).lower()
        src = src.replace('\n', '')
        words = word_token(text=src, language=language)
        src_words.append(words)
        src_new.append(src)

    for tar in tqdm(tar_all):
        tar = re.sub('\*\*', '', tar).lower()
        tar = tar.replace('\n','')
        words = word_token(text=tar, language=language)
        tar_words.append(words)
        tar_new.append(tar)

    src_all = src_new[:]
    tar_all = tar_new[:]
    dataset = []
    for src, tar, src_word, tar_word in zip(src_all, tar_all, src_words, tar_words):
        if len(src_word)<configure['max_content'] and len(tar_word)<configure['max_output'] and len(src_word)>0 and len(tar_word)>0:
            dataset.append([src, tar])

    print(len(dataset))
    with open('./data/dataset.pkl','wb') as f:
        pickle.dump(dataset,f)

    random.seed(2021)
    random.shuffle(dataset)
    dataset = dataset
    tot = len(dataset)
    with open('./data/all/all.txt', 'w', encoding='utf-8') as f:
        for data in dataset:
            f.write(data[0])
            f.write('\n')
            f.write(data[1])
            f.write('\n')

    for i, data in enumerate(dataset[:tot // 8*7]):
        with open('./data/train/{}.txt'.format(i+1), 'w', encoding='utf-8') as f:
            f.write(data[0])
            f.write('\n')
            f.write(data[1])

    for i, data in enumerate(dataset[tot // 8*7:]):
        with open('./data/test/{}.txt'.format(i+1), 'w', encoding='utf-8') as f:
            f.write(data[0])
            f.write('\n')
            f.write(data[1])


if __name__ == '__main__':
    main()
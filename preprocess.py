
import pickle
import torch
from tqdm import tqdm
import re
from util_common.nlp.word_dictionary import word2index
from util_common.nlp.word_token import word_token
from util_common.nlp.os import read_folder_content, read_content
import random
import numpy as np


def main():

    with open('./data/dataset-aligned.pkl','rb') as f:
        dataset_aligned = pickle.load(f)
    src_all = []
    tar_all = []
    for data in dataset_aligned:
        src_all.extend(data[0].split('。'))
        tar_all.extend(data[1].split('。'))
    language = 'chinese'
    src_words = []
    tar_words = []
    src_new = []
    tar_new = []
    configure = eval(read_content('configure'))
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
    length = []
    max_len = 64
    for src, tar, src_word, tar_word in zip(src_all, tar_all, src_words, tar_words):
        if len(src_word)<max_len and len(tar_word)<max_len and len(src_word)>0 and len(tar_word)>0:
            dataset.append([src, tar])
            length.append(len(src_word))

    arg_index = np.argsort(length)
    dataset_new = []
    for index in arg_index:
        dataset_new.append(dataset[index])
    dataset = dataset_new[:]
    print(len(dataset))
    cnt = 0
    for data in dataset:
        if data[0]!=data[1]:
            cnt += 1
    print(cnt)


    with open('./data/all/all.txt', 'w', encoding='utf-8') as f:
        for data in dataset:
            f.write(data[0])
            f.write('\n')
            f.write(data[1])
            f.write('\n')

    tot = len(dataset)
    with open('./data/train/src_txts.pkl', 'wb') as f:
        pickle.dump([u[0] for u in dataset[:tot // 8*7]], f)
    with open('./data/train/tar_txts.pkl', 'wb') as f:
        pickle.dump([u[1] for u in dataset[:tot // 8*7]], f)

    with open('./data/test/src_txts.pkl', 'wb') as f:
        pickle.dump([u[0] for u in dataset[tot // 8*7:]], f)
    with open('./data/test/tar_txts.pkl', 'wb') as f:
        pickle.dump([u[1] for u in dataset[tot // 8*7:]], f)



if __name__ == '__main__':
    main()
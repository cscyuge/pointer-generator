
import pickle
import torch
from tqdm import tqdm
import re
from util_common.nlp.word_dictionary import word2index
from util_common.nlp.word_token import word_token
from util_common.nlp.os import read_folder_content, read_content
import random
import numpy as np
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences


def main():
    with open('./data/dataset-aligned.pkl','rb') as f:
        dataset_aligned = pickle.load(f)
    src_all = []
    tar_all = []
    for data in dataset_aligned:
        src_all.extend(data[0].split('。'))
        tar_all.extend(data[1].split('。'))

    # bert_model = 'trueto/medbert-base-wwm-chinese'
    bert_model = 'hfl/chinese-bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    src_ids = []
    tar_ids = []
    len_cnt = [0 for _ in range(2600)]
    for src in tqdm(src_all):
        src = re.sub('\*\*', '', src).lower()
        tokens = tokenizer.tokenize(src)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        src_ids.append(ids)
        len_cnt[len(ids)] += 1

    for tar in tqdm(tar_all):
        tar = re.sub('\*\*', '', tar).lower()
        tokens = tokenizer.tokenize(tar)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        tar_ids.append(ids)
        len_cnt[len(ids)] += 1
    print(len(src_ids))
    src_ids_smaller = []
    tar_ids_smaller = []
    tar_txts = []
    max_len = 64
    for src, tar, txt in zip(src_ids, tar_ids, tar_all):
        if len(src)<max_len and len(tar)<max_len and len(src)>2 and len(tar)>2 :
            src_ids_smaller.append(src)
            tar_ids_smaller.append(tar)
            tar_txts.append(txt)
    src_ids = src_ids_smaller
    tar_ids = tar_ids_smaller
    print(len(src_ids))
    src_ids = np.array(src_ids)
    tar_ids = np.array(tar_ids)
    length = []
    for src in src_ids:
        length.append(len(src))
    arg_index = np.argsort(length)
    src_ids_unpad = src_ids[arg_index]
    tar_ids_unpad = tar_ids[arg_index]

    configure = eval(read_content('./configure'))
    src_ids = pad_sequences(src_ids, maxlen=configure['max_content'], dtype="long", value=0, truncating="post", padding="post")
    tar_ids = pad_sequences(tar_ids, maxlen=configure['max_output'], dtype="long", value=0, truncating="post", padding="post")

    src_masks = [[float(i != 0.0) for i in ii] for ii in src_ids]
    tar_masks = [[float(i != 0.0) for i in ii] for ii in tar_ids]
    src_ids = {'pad': src_ids, 'unpad': src_ids_unpad}
    tar_ids = {'pad': tar_ids, 'unpad': tar_ids_unpad}

    seq_len_srcs = len(src_ids['pad'][0]) - np.sum(src_ids['pad'] == 0, axis=1)
    arg_index = np.argsort(seq_len_srcs)
    src_ids['pad'] = np.array(src_ids['pad'])[arg_index]
    tar_ids['pad'] = np.array(tar_ids['pad'])[arg_index]
    tar_txts = np.array(tar_txts)[arg_index]

    with open('./data/train/src_ids.pkl', 'wb') as f:
        pickle.dump(src_ids['pad'][:len(src_ids['pad'])//10*9], f)
    with open('./data/train/src_masks.pkl', 'wb') as f:
        pickle.dump(src_masks[:len(src_ids['pad']) // 10 * 9], f)
    with open('./data/train/tar_ids.pkl', 'wb') as f:
        pickle.dump(tar_ids['pad'][:len(src_ids['pad'])//10*9], f)
    with open('./data/train/tar_masks.pkl', 'wb') as f:
        pickle.dump(tar_masks[:len(src_ids['pad']) // 10 * 9], f)

    with open('./data/test/src_ids.pkl', 'wb') as f:
        pickle.dump(src_ids['pad'][len(src_ids['pad'])//10*9:], f)
    with open('./data/test/src_masks.pkl', 'wb') as f:
        pickle.dump(src_masks[len(src_ids['pad'])//10*9:], f)
    with open('./data/test/tar_ids.pkl', 'wb') as f:
        pickle.dump(tar_ids['pad'][len(src_ids['pad'])//10*9:], f)
    with open('./data/test/tar_masks.pkl', 'wb') as f:
        pickle.dump(tar_masks[len(src_ids['pad'])//10*9:], f)
    with open('./data/test/tar_txts.pkl','wb') as f:
        pickle.dump(tar_txts[len(src_ids['pad'])//10*9:], f)




if __name__ == '__main__':
    main()
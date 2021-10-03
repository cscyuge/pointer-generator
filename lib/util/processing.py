
from util_common.nlp.word_dictionary import word_dictionary
from util_common.nlp.os import read_folder_content, read_content
from util_common.mongo.nlp import nlp_chinese_inter_content

from lib.dataset.generator import ChineseDataset

import torch

def read_configure_word_index(path):
    # Declare the hyperparameter
    configure = eval(read_content(path))

    # Get the word vs index
    word_index, index_word = word_dictionary(sentences=nlp_chinese_inter_content(configure["folder-all"]),
                                             language=configure["language"])
    return configure, word_index, index_word


def dataset_pipline(folder, batch_size, configure, shuffle, tokenizer):
    # Declare the dataset pipline
    dataset = ChineseDataset(folder)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle)
    return loader


def processing(path, tokenizer):
    # 1. Declare the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Declare the processing
    configure = eval(read_content(path))
    
    # 3. Declare the dataset pipline
    train_loader = dataset_pipline(folder=configure["folder-train"],
                                   batch_size=configure["batch_size"],
                                   configure=configure, shuffle=False, tokenizer=tokenizer)

    test_loader = dataset_pipline(folder=configure["folder-test"],
                                  batch_size=configure["batch_size"],
                                  configure=configure, shuffle=False, tokenizer=tokenizer)

    return device, configure, train_loader, test_loader
    

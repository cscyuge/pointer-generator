import pymongo
from util_ds.nlp.os import read_folder_content

def nlp_chinese_inter():
    client = pymongo.MongoClient("192.168.2.130",27017)
    IDs = []

    collection = client.nlp_chinese.inter

    for item in collection.find():
        id = item["_id"]
        IDs.append(id)
    return IDs, collection


def nlp_chinese_inter_content(path):
    # client = pymongo.MongoClient("192.168.2.130",27017)
    # sentences = []
    
    # collection = client.nlp_chinese.inter

    # for item in collection.find():
    #     input = item["input"]
    #     target = item["target"]
    #     sentences.append(input + " " + target)
    return read_folder_content(path)
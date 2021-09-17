
from util_ds.nlp.os import read_folder_content
import pickle

def main():
    test_data = read_folder_content('./cache/test/')
    print(len(test_data))
    with open('./cache/test_tar.pkl','wb') as f:
        pickle.dump(test_data,f)


if __name__ == '__main__':
    main()
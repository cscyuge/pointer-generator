
from lib.util.processing import processing
from lib.model.encoder import SimpleEncoder
from lib.model.decoder import AttentionDecoder

import torch
import os
import random
import time
from importlib import  import_module
import pickle
from transformers import BertTokenizer
from bleu_eval import count_score


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    bert_model = 'hfl/chinese-bert-wwm-ext'

    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # 1. Declare the hyperparameter
    device, configure, train_loader, test_loader, valid_loader = processing("./configure", tokenizer)

    # print(len(word_index))
    print(tokenizer.vocab_size)

    # Declare the encoder model
    model_encoder = SimpleEncoder(configure).to(device)
    model_decoder = AttentionDecoder(configure, device).to(device)
    model_encoder.load_state_dict(torch.load('./cache/model_encoder_best.ckpt'))
    model_decoder.load_state_dict(torch.load('./cache/model_decoder_best.ckpt'))

    # Define the optimizer and loss
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss(reduce=False)
    # encoder optimizer
    optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=configure["lr"])
    optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=configure["lr"])

    start_time = time.clock()
    max_bleu = 0


    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0

        results = []
        tar_txts = []
        for idx, item in enumerate(test_loader):

            # transfer to long tensor
            input, target = [i.type(torch.LongTensor).to(device) for i in item[:2]]
            tar_txts.extend(item[-1])

            # input = input.cuda()
            # target = target.cuda()

            batch_size = input.size(0)
            # Encoder
            encoder_out, encoder_hidden = model_encoder(input)

            # Decoder
            # declare the first input <go>
            decoder_input = torch.tensor([tokenizer.cls_token_id ] *batch_size,
                                         dtype=torch.long, device=device).view(batch_size, -1)
            decoder_hidden = encoder_hidden
            seq_loss = 0
            z = torch.ones([batch_size ,1 ,configure["hidden_size"]]).to(device)
            coverage = torch.zeros([batch_size ,configure["max_content"]]).to(device)
            result = []
            sentences = [[] for _ in range(batch_size)]
            for i in range(configure["max_output"]):
                decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, input, coverage)


                _, decoder_input = torch.max(decoder_output, 1)
                decoder_input = decoder_input.view(batch_size, -1)


                decoder_hidden = decoder_hidden

                total += batch_size
                correct += (torch.max(decoder_output, 1)[1] == target[: ,i]).sum().item()
                # print(torch.max(decoder_output, 1)[1],target[:,i])
                symbols = torch.max(decoder_output, 1)[1].cpu().tolist()
                for i, symbol in enumerate(symbols):
                    sentences[i].append(symbol)

            sentences = [tokenizer.convert_ids_to_tokens(u) for u in sentences]
            results.extend([''.join(_) for _ in sentences])

        with open("./result/sentence_out.pkl", "wb") as a:
            pickle.dump([_.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '') +'\n' for _ in results],a)
        with open("./result/sentence_ref.pkl", "wb") as a:
            pickle.dump(tar_txts, a)

        references = [[u] for u in tar_txts[:len(results)]]
        results = [u.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '') for u in results]
        # print(len(references),len(results))
        bleu = count_score(results, references, tokenizer)
        print("BLEU: ", bleu)




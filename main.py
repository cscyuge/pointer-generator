

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    bert_model = 'hfl/chinese-bert-wwm-ext'

    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # 1. Declare the hyperparameter
    device, configure, train_loader, test_loader, valid_loader = processing("./configure", tokenizer)

    # print(len(word_index))
    print(tokenizer.vocab_size)

    # Declare the encoder model
    model_encoder = SimpleEncoder(configure).to(device)
    model_decoder = AttentionDecoder(configure, device).to(device)


    # Define the optimizer and loss
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.NLLLoss(reduce=False)
    # encoder optimizer
    optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=configure["lr"])
    optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=configure["lr"])

    start_time = time.clock()
    max_bleu = 0
    # Training
    for epoch in range(configure["epochs"]):
        for idx, item in enumerate(train_loader):

            # transfer to long tensor
            input, target = [i.type(torch.LongTensor).to(device) for i in item[:2]]


            # if input.size(0) != configure["batch_size"]: continue
            batch_size = input.size(0)
            # Encoder
            encoder_out, encoder_hidden = model_encoder(input)

            # Decoder
            decoder_input = torch.tensor([tokenizer.cls_token_id]*batch_size,
                                         dtype=torch.long, device=device).view(batch_size, -1)
            decoder_hidden = encoder_hidden
            z = torch.ones([batch_size,1,configure["hidden_size"]]).to(device)
            coverage = torch.zeros([batch_size,configure["max_content"]]).to(device)
            seq_loss = 0
            for i in range(configure["max_output"]):

                decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, input, coverage)

                coverage = coverage

                if random.randint(1, 10) > 5:
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(batch_size, -1)
                else:
                    decoder_input = target[:,i].view(batch_size, -1)

                decoder_hidden = decoder_hidden

                step_coverage_loss = torch.sum(torch.min(attn.reshape(-1,1), coverage.reshape(-1,1)), 1)
                step_coverage_loss = torch.sum(step_coverage_loss)

                seq_loss += (criterion(decoder_output, target[:,i]))

                # print(seq_loss)

                seq_loss += step_coverage_loss

                # print(decoder_input)
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            seq_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            if (idx) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Coverage Loss: {:4f} , Time cost: {:4f}'
                    .format(epoch+1, configure["epochs"], idx, len(train_loader), seq_loss.item()/batch_size/configure['max_output'],
                            step_coverage_loss.item(),time.clock()-start_time))
                start_time = time.clock()


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
                decoder_input = torch.tensor([tokenizer.cls_token_id]*batch_size,
                                            dtype=torch.long, device=device).view(batch_size, -1)
                decoder_hidden = encoder_hidden
                seq_loss = 0
                z = torch.ones([batch_size,1,configure["hidden_size"]]).to(device)
                coverage = torch.zeros([batch_size,configure["max_content"]]).to(device)
                result = []
                sentences = [[] for _ in range(batch_size)]
                for i in range(configure["max_output"]):
                    decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, input, coverage)

    
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(batch_size, -1)


                    decoder_hidden = decoder_hidden

                    total += batch_size
                    correct += (torch.max(decoder_output, 1)[1] == target[:,i]).sum().item()
                    # print(torch.max(decoder_output, 1)[1],target[:,i])
                    symbols = torch.max(decoder_output, 1)[1].cpu().tolist()
                    for i, symbol in enumerate(symbols):
                        sentences[i].append(symbol)

                sentences = [tokenizer.convert_ids_to_tokens(u) for u in sentences]
                results.extend([''.join(_) for _ in sentences])

            with open("./result/test{}.txt".format(epoch), "w", encoding="utf-8") as a:
                a.writelines([_+'\n' for _ in results])
            references = [[u] for u in tar_txts[:len(results)]]
            results = [u.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '') for u in results]
            # print(len(references),len(results))
            bleu = count_score(results, references, tokenizer)
            print("BLEU: ", bleu)
            if bleu>max_bleu:
                max_bleu = bleu
                torch.save(model_encoder.state_dict(), './cache/model_encoder_best.ckpt')
                torch.save(model_decoder.state_dict(), './cache/model_decoder_best.ckpt')

            if epoch % 10==0:
                torch.save(model_encoder.state_dict(), './cache/model_encoder_{}.ckpt'.format(epoch))
                torch.save(model_decoder.state_dict(), './cache/model_decoder_{}.ckpt'.format(epoch))

            # print('Test Accuracy of the model on the test: {} %'.format(acc))

    torch.save(model_encoder.state_dict(), './cache/model_encoder_final.ckpt')
    torch.save(model_decoder.state_dict(), './cache/model_decoder_final.ckpt')



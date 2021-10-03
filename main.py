

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    bert_model = 'hfl/chinese-bert-wwm-ext'

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    with open('./data/test/tar_txts.pkl','rb') as f:
        tar_txts = pickle.load(f)

    # 1. Declare the hyperparameter
    device, configure, train_loader, test_loader = processing("./configure", tokenizer)

    # print(len(word_index))
    print(tokenizer.vocab_size)

    # Declare the encoder model
    model_encoder = SimpleEncoder(configure).to(device)
    model_decoder = AttentionDecoder(configure, device).to(device)


    # Define the optimizer and loss
    criterion = torch.nn.CrossEntropyLoss()
    # encoder optimizer
    optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=configure["lr"])
    optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=configure["lr"])

    start_time = time.clock()
    max_bleu = 0
    # Training
    for epoch in range(configure["epochs"]):
        for idx, item in enumerate(train_loader):

            # transfer to long tensor
            input, target = [i.type(torch.LongTensor).to(device) for i in item]


            if input.size(0) != configure["batch_size"]: continue
            # Encoder   
            encoder_out, encoder_hidden = model_encoder(input)
            
            # Decoder
            decoder_input = torch.tensor([tokenizer.cls_token_id]*configure["batch_size"],
                                         dtype=torch.long, device=device).view(configure["batch_size"], -1)
            decoder_hidden = encoder_hidden
            z = torch.ones([configure["batch_size"],1,configure["hidden_size"]]).to(device)
            coverage = torch.zeros([configure["batch_size"],configure["max_content"]]).to(device)
            seq_loss = 0
            for i in range(configure["max_output"]):

                decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, input, coverage)

                coverage = coverage

                if random.randint(1, 10) > 5:
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(configure["batch_size"], -1)
                else:
                    decoder_input = target[:,i].view(configure["batch_size"], -1)

                decoder_hidden = decoder_hidden

                step_coverage_loss = torch.sum(torch.min(attn.reshape(-1,1), coverage.reshape(-1,1)), 1) 
                step_coverage_loss = torch.sum(step_coverage_loss)
                # print(coverage)
                # print("---")
                # decoder_output = decoder_output.reshape(configure["batch_size"], -1, 1)
                # print(step_coverage_loss)
                # print((criterion(decoder_output, target[:,i].reshape(configure["batch_size"],-1))))
                # print(-torch.log(decoder_output+target[:,i]))
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
                    .format(epoch+1, configure["epochs"], idx, len(train_loader), seq_loss.item()/configure['batch_size'],
                            step_coverage_loss.item(),time.clock()-start_time))
                start_time = time.clock()


        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0

            results = []
            for idx, item in enumerate(test_loader):
                
                # transfer to long tensor
                input, target = [i.type(torch.LongTensor).to(device) for i in item]

                # input = input.cuda()
                # target = target.cuda()

                if input.size(0) != configure["batch_size"]: continue
                # Encoder   
                encoder_out, encoder_hidden = model_encoder(input)
                
                # Decoder 
                # declare the first input <go>
                decoder_input = torch.tensor([tokenizer.cls_token_id]*configure["batch_size"],
                                            dtype=torch.long, device=device).view(configure["batch_size"], -1)
                decoder_hidden = encoder_hidden
                seq_loss = 0
                z = torch.ones([configure["batch_size"],1,configure["hidden_size"]]).to(device)
                coverage = torch.zeros([configure["batch_size"],configure["max_content"]]).to(device)
                result = []
                sentences = [[] for _ in range(configure['batch_size'])]
                for i in range(configure["max_output"]):
                    decoder_output, decoder_hidden, z, attn, coverage = model_decoder(decoder_input, decoder_hidden, encoder_out, z, input, coverage)

    
                    _, decoder_input = torch.max(decoder_output, 1)
                    decoder_input = decoder_input.view(configure["batch_size"], -1)


                    decoder_hidden = decoder_hidden

                    total += configure["batch_size"]
                    correct += (torch.max(decoder_output, 1)[1] == target[:,i]).sum().item()
                    # print(torch.max(decoder_output, 1)[1],target[:,i])
                    symbols = torch.max(decoder_output, 1)[1].cpu().tolist()
                    for i, symbol in enumerate(symbols):
                        sentences[i].append(symbol)

                sentences = [tokenizer.convert_ids_to_tokens(u) for u in sentences]
                results.extend([''.join(_) for _ in sentences])

            with open("./result/test{}.txt".format(epoch), "w", encoding="utf-8") as a:
                a.writelines([_+'\n' for _ in results])
            references = [[u] for u in tar_txts]
            results = [u.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '') for u in results]
            bleu = count_score(results, references, tokenizer)
            print("BLEU: ", bleu)
            if bleu>max_bleu:
                max_bleu = bleu
                torch.save(model_encoder.state_dict(), './cache/model_encoder_best.ckpt')
                torch.save(model_decoder.state_dict(), './cache/model_decoder_best.ckpt')


            acc = 100 * correct / total
            if epoch % 10==0:
                torch.save(model_encoder.state_dict(), './cache/model_encoder_{}.ckpt'.format(epoch))
                torch.save(model_decoder.state_dict(), './cache/model_decoder_{}.ckpt'.format(epoch))

            # print('Test Accuracy of the model on the test: {} %'.format(acc))

    torch.save(model_encoder.state_dict(), './cache/model_encoder_final.ckpt')
    torch.save(model_decoder.state_dict(), './cache/model_decoder_final.ckpt')



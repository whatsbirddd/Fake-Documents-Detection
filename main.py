import torch
import argparse
from train import flat_accuracy, train, predict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import time
import datetime
import nsml
from nsml import DATASET_PATH, DATASET_NAME
import os
from transformers import ElectraModel, ElectraTokenizer, ElectraForSequenceClassification, AutoTokenizer
from util import mixup
import re
from model import KRElectraClassificationModel, BertClassificationModel, BigBirdClassificationModel, KRElectraLstmClassifier, BertLstmClassifier, BigBirdLstmClassifier
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig

def generate_data_loader(file_path, tokenizer, args):
    def get_input_ids(data): #input : list - str
        document_bert = ["[CLS] " + str(s) + " [SEP]" for s in data]
        tokenized_texts = [tokenizer.tokenize(s) for s in document_bert]
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=args.maxlen, dtype='long', truncating='post', padding='post')
        return input_ids

    def get_attention_masks(input_ids):
        attention_masks = []
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks

    def get_data_loader(inputs, masks, labels, batch_size=args.batch):
        data = TensorDataset(torch.tensor(inputs), torch.tensor(masks), torch.tensor(labels))
        sampler = RandomSampler(data) if args.mode == 'train' else SequentialSampler(data)
        data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return data_loader

    data_df = pd.read_csv(file_path)
    
    #mixed data추가
    if file_path == args.train_path:
        mixed_df = pd.read_csv('./mixed_data.csv')
        data_df = data_df.append(mixed_df)
    print("data 개수",len(data_df))    
    input_ids = get_input_ids(data_df['contents'].values)
    attention_masks = get_attention_masks(input_ids)
    data_loader = get_data_loader(input_ids, attention_masks, data_df['label'].values if args.mode=='train' else [-1]*len(data_df))

    return data_loader


def bind_nsml(model, args=None):
    def save(dir_name, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))

    def load(dir_name, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'), map_location=args.device)
        model.load_state_dict(state, strict=False)
        print('model is loaded')

    def infer(file_path, **kwargs):
        print('start inference')
        tokenizer = ElectraTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator")
        test_dataloader = generate_data_loader(file_path, tokenizer, args)
        results, _ = predict(model, args, test_dataloader)
        return results

    nsml.bind(save, load, infer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
    parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--wd", type=float, default=0) #weight decay - 0,0.1,0.01
    parser.add_argument("--warmup", type=int, default=0) #warm up steps - 0, 100, 500

    args = parser.parse_args()

    # initialize args
    args.train_path = os.path.join(DATASET_PATH, args.train_path)
    args.valid_path = os.path.join(DATASET_PATH, args.valid_path)

    print(args)

    # model load
    #model = KRElectraClassificationModel(args, embed_dim=768, num_labels=2)
    model = KRElectraLstmClassifier(
                        hidden_size=64,  #128 or 64
                        output_size=2,  #loss function 바꿔서 output_size=1
                        embed_dim=768,
                        num_layers=3, 
                        batch_first=True, 
                        bidirectional=True,
                        maxlen = args.maxlen)
    model.to(args.device)
    bind_nsml(model, args=args)

    #추가 학습
    #nsml.load(checkpoint='3', session='KR96327/airush2022-1-2a/24')
    #nsml.save('saved')
    #model.to(args.device)
    print("모델 로드 완료")

    # test mode
    if args.pause:
        nsml.paused(scope=locals())

    # train mode
    if args.mode == "train":
        tokenizer = ElectraTokenizer.from_pretrained("snunlp/KR-ELECTRA-discriminator")
        #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        train_dataloader = generate_data_loader(args.train_path, tokenizer, args)
        validation_dataloader = generate_data_loader(args.valid_path, tokenizer, args)
        train(model, args, train_dataloader, validation_dataloader)

    if args.mode == "mixup":
        print("mixup start")
        data_df = pd.read_csv(args.train_path)
        sample = data_df[data_df['label'] == 0][:12500]
        target = data_df[data_df['label'] == 0][-4000:]
        mixed_df = pd.DataFrame(columns=['contents','label'])
        
        print("mixup dataset")
        mixed_df = mixup(sample,target,50000)
        
        #save mixed-dataset
        print("save dataset")
        folder_path = './data'
        os.makedirs(folder_path, exist_ok=True)
        mixed_df.to_csv(f'{folder_path}/mixed_data.csv', index=False)
        nsml.save_folder('mixed_data', folder_path)

        
        
        
        
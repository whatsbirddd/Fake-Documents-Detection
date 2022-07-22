from cProfile import label
from typing import Counter
import torch
from torch import nn
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import nsml
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch import nn
from util import FocalLoss


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    

def predict(model, args, data_loader):
    print('start predict')
    model.eval()

    eval_accuracy = []
    logits = []
    
    for step, batch in enumerate(data_loader):
        batch = tuple(t.to(args.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        logit = outputs

        logit = logit.detach().cpu().numpy()
        label = b_labels.cpu().numpy()

        logits.append(logit)

        accuracy = flat_accuracy(logit, label)
        eval_accuracy.append(accuracy)

    logits = np.vstack(logits)
    predict_labels = np.argmax(logits, axis=1)
    return predict_labels, np.mean(eval_accuracy)

def train(model, args, train_loader, valid_loader):
    #criterion = FocalLoss()
    class_weights=torch.tensor([1,0.03],dtype=torch.float,device=args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      eps=args.eps,
                      weight_decay = args.wd
                      )
    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup, 
                                                num_training_steps=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print('start training')
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for step, batch in enumerate(train_loader):
            model.zero_grad()
            batch = tuple(t.to(args.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            logits = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
            
            loss = criterion(logits,b_labels)
            train_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        #evaluation
        print("#########################################")
        print("Epoch {0}, train_loss {1}".format(epoch, train_loss))
        print("#########################################")
        
        avg_train_loss = np.mean(train_loss)
        _, avg_train_accuracy = predict(model, args, train_loader)
        val_preds, avg_val_accuracy = predict(model, args, valid_loader)
        
        print("Epoch {0},  Average training loss: {1:.8f} , Average accuracy: {2:.8f},Validation accuracy : {3:.8f}"\
            .format(epoch, avg_train_loss, avg_train_accuracy, avg_val_accuracy))
        
        data_df = pd.read_csv(args.valid_path)
        y_true = data_df["label"].values
        print(confusion_matrix(y_true, val_preds))
        
        nsml.save(epoch)
    return model

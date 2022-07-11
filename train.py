import torch
from torch import nn

from model import KRSentenceBERT
from loss import BiEncoderNllLoss

import torch.optim as optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch_optimizer as custom_optim

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from utils import * 

####################################################################
#
# set model /criterion/ optimizer
#
####################################################################

def get_model(config):
    model = KRSentenceBERT()
    return model
    
def get_crit(config):
    criterion = BiEncoderNllLoss()    
    return criterion

def get_optimizer(config, model):
    scaler = torch.cuda.amp.GradScaler()
    optimizer = custom_optim.RAdam(model.parameters(), lr=config.learning_rate)
    
    return optimizer, scaler

def initiate(config, train_loader, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" device : {device}")
    model = get_model(config).to(device)
    optimizer, scaler = get_optimizer(config, model)
    criterion = get_crit(config)

    settings = {'model': model,
                'scaler' : scaler,
                'optimizer': optimizer,
                'criterion': criterion}
    
    return train_model(settings, config, train_loader,test_loader, scaler)


####################################################################
#
# training and evaluation scripts
#
####################################################################

def train_model(settings, config, train_loader,  test_loader, scaler):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    def train(epoch, model, optimizer, criterion, scaler):
        train_loss = 0.0
        total_pred, total_label = [], []
        num_batches = config.n_train // config.batch_size 
        
        model.train()
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                abstract = batch_to_device(batch['abstract'], device)
                paragraph = batch_to_device(batch['paragraph'], device)
                
                # print(f"abstract: {abstract['input_ids'].shape}") # torch.Size([64, 128]) 

                optimizer.zero_grad()            
                with torch.cuda.amp.autocast():
                    abstract_emb, paragraph_emb = model(abstract, paragraph)
                    loss, preds = criterion.calc(abstract_emb, paragraph_emb)
                
                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss

                preds = preds.detach().cpu().numpy().tolist()
                label = list(range(config.batch_size))

                # append- epoch-wise
                total_pred += preds
                total_label +=  label

                # batch_accuracy = accuracy_score(preds, label)
                # print(f"batch_accuracy : {batch_accuracy}")
        accuracy = accuracy_score(total_label, total_pred)
        f1_score_ = f1_score(total_label, total_pred, average = 'macro')
        print(f'[Epoch {epoch} training ] : loss = {float(train_loss/num_batches) :.4f}, accuracy = {accuracy:.4f}, f1_score = {f1_score_:.4f}')
                            
        return accuracy, train_loss/num_batches
            
 
    def valid(epoch, model, criterion, test=False):
        valid_loss = 0.0
        total_pred, total_label = [], []
        num_batches = config.n_test // config.batch_size

        model.eval()           

        with torch.no_grad():
            with tqdm(test_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    abstract = batch_to_device(batch['abstract'], device)
                    paragraph = batch_to_device(batch['paragraph'], device)

                    optimizer.zero_grad()            
                    with torch.cuda.amp.autocast():
                        abstract_emb, paragraph_emb = model(abstract, paragraph)
                        loss, preds = criterion.calc(abstract_emb, paragraph_emb)
                    
                    valid_loss += loss

                    preds = preds.detach().cpu().numpy().tolist()
                    label = list(range(config.batch_size))
                    # append- epoch-wise
                    total_pred += preds
                    total_label +=  label

            accuracy = accuracy_score(total_label, total_pred)
            f1_score_ = f1_score(total_label, total_pred, average = 'macro')
            print(f'[Epoch {epoch} testing ] : loss = {float(valid_loss/num_batches) :.4f}, accuracy = {accuracy:.4f}, f1_score = {f1_score_:.4f}')
                                
            return accuracy, valid_loss/num_batches
                

#################################################################################
#                Let's Start training / validating / testing
#################################################################################


    for epoch in range(1, config.num_epochs+1):
        train(epoch, model, optimizer, criterion, scaler) # trian
        # valid(epoch, model, criterion, test=False) # valid
        valid(epoch, model, criterion, test=True) # test

        if (epoch%5==0) and (epoch >=5):
            save_model(config, epoch, model)


if __name__== "__main__":
     device = 'cuda' if torch.cuda.is_available() else 'cpu'
     print(device)
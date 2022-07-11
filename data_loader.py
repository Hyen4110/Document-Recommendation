import numpy as np
import pandas as pd
import pickle
from regex import E
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
from sentence_transformers import SentenceTransformer


class PaperDataset(Dataset):
    '''
    data(dataframe)
    | --- |  doc_id     | abstract_full     | paragraph_full   |  
    |  1  |  AF_000122  |    이 논문은 ~~    |   선행 연구는 ~   |
    '''
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.tokenizer = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS').tokenizer
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):          
        abstract_full = self.text_to_inputids(self.data['abstract_full'].loc[idx]) 
        paragraph_full = self.text_to_inputids(self.data['paragraph_full'].loc[idx])
                
        return {'abstract': abstract_full, 'paragraph': paragraph_full}

    def text_to_inputids(self, text):
        inputs = self.tokenizer(text, 
                        padding='max_length', 
                        truncation=True, 
                        return_tensors="pt", 
                        max_length = self.config.max_token_len)
        
        token_ids = torch.squeeze(inputs['input_ids']) # tensor type
        attention_mask = torch.squeeze(inputs['attention_mask']) # tensor type
        token_type_ids = torch.squeeze(inputs['token_type_ids']) # tensor type
        
        return {"input_ids":token_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}
        # return [token_ids, attention_mask, token_type_ids]
    
def load_data(config):
    data = pd.read_csv(config.data_path, index_col = 0)
    print(data.info())
    return data

# split into train/valid/test
def get_data_split(config, test = True):
    data = load_data(config)
    # data = data[:10000]
    train_size = int(len(data)*0.7)
    print(f"train_size: {train_size}")
    
    random_index = torch.randperm(len(data))
    train_df = data.loc[random_index][:train_size]
    train_df.reset_index(drop=True, inplace= True)

    test_df = data.loc[random_index][train_size:]
    test_df.reset_index(drop=True, inplace= True)

    return train_df, test_df
        
def get_loaders(config):
    train_df, test_df = get_data_split(config)
    train_loader = DataLoader( dataset = PaperDataset( config, train_df),
                                batch_size = config.batch_size,
                                shuffle=True,
                                drop_last = True)
    test_loader = DataLoader( dataset = PaperDataset( config, test_df),
                                batch_size=config.batch_size,
                                shuffle=False,
                                drop_last = True)
                                
    config.n_train, config.n_test =len(train_loader.dataset), len(test_loader.dataset)
    print(f" [data counts] train_loader: {len(train_loader.dataset)},test_loader : {len(test_loader.dataset)}")

    return train_loader, test_loader


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Kibo Document Recommenda (Pytorch)')
    parser.add_argument('--batch_size', type=int, default = 3)
    parser.add_argument('--max_token_len', type=int, default = 128)

    parser.add_argument('--data_path', type=str, default = './data/AIHub/paper_summary/train/paper/csv/paper_train_total_2.csv') 
    parser.add_argument('--n_train', type=int, default = 0) # not hyper-parameter
    parser.add_argument('--n_test', type=int, default = 0) # not hyper-parameter 
    config = parser.parse_args()


    train_loader, test_loader = get_loaders(config)
    print("=====================train_loader data sample [0]")
    # ab, para = next(iter(train_loader))
    # print("abstract",ab.shape)
    # print("paragraph ", para.shape)
    print("abstract", next(iter(train_loader))['abstract'])
    print("paragraph",next(iter(train_loader))['paragraph'])

    pad_idx = next(iter(train_loader))['abstract']['attention_mask'].numpy()
    # print(np.argmin(pad_idx)) #172

    features = next(iter(train_loader))['abstract']
    print(f"features shape: {features['input_ids'].shape}")
    trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask'], 'token_type_ids': features['token_type_ids']}
    print(f"trans_features: {trans_features}")
    print(features['input_ids'].shape)
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    # a = model (token_ids, attention_mask, token_type_ids)
    print(model(features)['sentence_embedding'])
    # tokenizer = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS').tokenizer
    # inputs = tokenizer(["안녕하세요. 지금 제가 어디에 있나요?","안녕하세요. 지금 제가 어디에 있나요?"], 
    #                     padding='max_length', 
    #                     truncation=True, 
    #                     return_tensors="pt", 
    #                     max_length = config.max_token_len) 
    # print(inputs)
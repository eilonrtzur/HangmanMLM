import pickle
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import check_random_state
from torch.autograd import Variable
from itertools import product
from create_data import TrainData
from train_models import Head, MultiHeadAttention, Block, FeedFoward
from train_models import PositionalEncoding, TransformerHangman, load_dataset
import time

torch.set_printoptions(sci_mode=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pad_size = 20
num_neurons = 16
dropout=0.2
learning_rate = 0.002
batch_size = 128
max_iterations = 50

def load_models(save_files: list[str]) -> list[TransformerHangman]:
    """ loads trained models """
    models = []
    for save_file in save_files:
        with open(os.path.join('models', save_file),"rb") as file:
            models.append(pickle.load(file))
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return models

def length_of_batch_words(batched_words: torch.Tensor) -> list[int]:
    """ creates a list of integers representing the length of each word in batch before padding """
    length_of_words = [0]*batched_words.size(0)
    for i in range(len(length_of_words)):
        j = 0
        while batched_words[i,j]!=27:
            j+=1
        length_of_words[i] = j
    return length_of_words

def feed_model(model: TransformerHangman, model_block_size: int, batched_encoded_words: torch.Tensor, mode: str | None = None) -> torch.Tensor:
    """ evaluates component model on padded word, outputs logits for each position """
    model.eval()
    length_of_words = length_of_batch_words(batched_encoded_words)
    output_tensor = torch.zeros(batched_encoded_words.size(0),batched_encoded_words.size(1),28)
    batched_encoded_words = torch.unsqueeze(batched_encoded_words,1)
    for i in range(len(length_of_words)):
        if length_of_words[i]<model_block_size:
            continue  
        else:
            if mode == 'prefix':
                output_tensor[i,:3,:-1] += torch.squeeze(model(batched_encoded_words[i,:,:3]))
            elif mode == 'suffix':
                output_tensor[i,length_of_words[i]-3:length_of_words[i],:-1] += torch.squeeze(model(batched_encoded_words[i,:,length_of_words[i]-3:length_of_words[i]]))
            else:
                for j in range(length_of_words[i] - model_block_size + 1):
                    output_tensor[i,j:j+model_block_size,:-1] += torch.squeeze(model(batched_encoded_words[i,:,j:j+model_block_size]))
                for k in range(length_of_words[i]):
                    output_tensor[i,k,:-1] = output_tensor[i,k,:-1]/min(model_block_size, min(k+1,length_of_words[i]-k)) # dividing by how many times that position was predicted
    return output_tensor

def batch_masks(length_of_words: list[int]) -> list[torch.Tensor]:
    """ creates masks for word avoiding padding """
    masks = torch.ones(len(length_of_words),pad_size)
    reverse_masks = torch.zeros(len(length_of_words),pad_size)
    for i in range(len(length_of_words)):
        mask = torch.randint(0,2,(1,length_of_words[i]))
        while torch.sum(mask) == length_of_words[i]: # must make sure we mask something
            mask = torch.randint(0,2,(1,length_of_words[i]))
        masks[i,:length_of_words[i]]=mask
        reverse_masks[i,:length_of_words[i]] = torch.ones_like(mask) - mask
    return masks, reverse_masks

def models_outputs(models: list[TransformerHangman], x: torch.Tensor) -> torch.Tensor:
    """ evaluates all component models and combines into one tensor """
    logits = []
    logits.append(feed_model(models[0],3,x,'prefix'))
    logits.append(feed_model(models[1],3,x,'suffix'))
    for i in range(2,8):
        logits.append(feed_model(models[i],i+1,x))
    logits = torch.stack(logits, dim = 2)
    return logits

class Head_MoE(nn.Module):
    """ one head of MoE corresponding to a specific position in word """
    def __init__(self, num_neurons: int = 16, num_components: int = 8, pad_size : int = 20, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pad_size, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_components), 
            nn.Dropout(dropout),
        )
       
    def forward(self, x: torch.Tensor):
        weights = self.net(x)
        weights = F.softmax(weights, dim = -1)
        return weights

class MoE(nn.Module):
    """ very basic Mixture of Experts by stacking individual MoE heads """
    def __init__(self, num_neurons: int = 16, num_components: int = 8, pad_size : int = 20, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head_MoE(num_neurons, num_components, pad_size, dropout) for _ in range(pad_size)])
        self.dropout = nn.Dropout(dropout)
       
    def forward(self, x: torch.Tensor):
        out = torch.stack([h(x) for h in self.heads], dim=1) # out size = batch_size x pad_size x num_components
        out = self.dropout(out)
        return out

def train(model: MoE, component_models: list[TransformerHangman],criterion: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epochs: int) -> None:
    """ training loop for mixture of experts """
    for epoch in range(epochs):
        total = 0
        for x, y in train_loader:
            masks, reverse_masks = batch_masks(length_of_batch_words(x)) 
            x = torch.mul(x.float(),masks)
            optimizer.zero_grad()
            weights = model(x)
            logits = models_outputs(component_models, x)
            weights = weights[:,:,None,:]
            outputs = torch.matmul(weights,logits).squeeze()
            outputs = outputs.permute(0,2,1)
            y = torch.mul(y,reverse_masks).long()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total+=loss.item() #cumulative loss
        if epoch % 25 == 0:
            print('Epoch:', epoch, 'Loss:', total)

def train_and_save_MoE(model_files: list[str], dataset_file: str, num_neurons: int, max_iter: int, save_file: str) -> None:
    """ trains and saves the MoE """
    model = MoE(num_neurons, len(model_files), pad_size, dropout)
    model = model.to(device)
    component_models = load_models(model_files)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    train_loader = DataLoader(load_dataset(dataset_file),batch_size,shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model,component_models,criterion,train_loader,optimizer,max_iter)
    with open(os.path.join('models', save_file), "wb") as file: pickle.dump(model, file)
    return

if __name__ == '__main__':
    """ trains the MoE model """
    model_files = ['prefix_model.pkl', 'suffix_model.pkl', '3gram_model.pkl', '4gram_model.pkl', '5gram_model.pkl', '6gram_model.pkl', '7gram_model.pkl', '8gram_model.pkl']
    train_and_save_MoE(model_files,'padded_words.pkl',num_neurons,max_iterations,'MoE_new.pkl')
import pickle
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import check_random_state
from torch.autograd import Variable
from itertools import product
from CreateData import TrainData
from TrainModels import Head, MultiHeadAttention, Block, FeedFoward
from TrainModels import PositionalEncoding, TransformerHangman, load_dataset
import time

torch.set_printoptions(sci_mode=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pad_size = 20
num_neurons = 16
dropout=0.2
learning_rate = 0.002
batch_size = 128
max_iterations = 200

def load_models(save_files: list[str]) -> list[TransformerHangman]:
    """ loads trained models """
    models = []
    for save_file in save_files:
        file = open(os.path.join('models', save_file),"rb")
        models.append(pickle.load(file))
    for model in models:
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

def feed_model(model: TransformerHangman, model_block_size: int, batched_encoded_words: torch.Tensor, mode: str = None) -> torch.Tensor:
    """ evaluates component model on padded word, outputs logits for each position """
    model.eval()
    length_of_words = length_of_batch_words(batched_encoded_words)
    output_tensor = torch.zeros(batched_encoded_words.size(0),batched_encoded_words.size(1),28)
    # output_tensor[:,:,27] = float('-inf')
    batched_encoded_words = torch.unsqueeze(batched_encoded_words,1)
    for i in range(len(length_of_words)):
        if length_of_words[i]<model_block_size:
            # output_tensor[i,:,:] = float('-inf')
            continue
            
        else:
            if mode == 'prefix':
                # output_tensor[i,3:,:] = float('-inf')
                output_tensor[i,:3,:-1] += torch.squeeze(model(batched_encoded_words[i,:,:3]))

            elif mode == 'suffix':
                # output_tensor[i,:,:] = float('-inf')
                output_tensor[i,length_of_words[i]-3:length_of_words[i],:-1] += torch.squeeze(model(batched_encoded_words[i,:,length_of_words[i]-3:length_of_words[i]]))

            else:
                # output_tensor[i,length_of_words[i]:,:] = float('-inf')
                for j in range(length_of_words[i] - model_block_size + 1):
                    output_tensor[i,j:j+model_block_size,:-1] += torch.squeeze(model(batched_encoded_words[i,:,j:j+model_block_size]))
 
    return output_tensor

def batch_masks(length_of_words: list[int]) -> torch.Tensor:
    """ creates masks for word avoiding padding """
    masks = torch.ones(len(length_of_words),pad_size)
    for i in range(len(length_of_words)):
        mask = torch.randint(0,2,(1,length_of_words[i]))
        while torch.sum(mask) == length_of_words[i]: # must make sure we mask something
            mask = torch.randint(0,2,(1,length_of_words[i]))
        masks[i,:length_of_words[i]]=mask
    return masks

def models_outputs(models: list[TransformerHangman], x: torch.Tensor) -> torch.Tensor:
    """ evaluates all component models and combines into one tensor """
    logits = []
    logits.append(feed_model(models[0].eval(),3,x,'prefix'))
    logits.append(feed_model(models[1].eval(),3,x,'suffix'))
    for i in range(2,8):
        logits.append(feed_model(models[i].eval(),i+1,x))
    logits = torch.stack(logits)
    logits = logits.permute(1,2,0,3)
    return logits

class MoE(nn.Module):
    """ very basic Mixture of Experts """
    def __init__(self, num_neurons: int, num_components: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_components), 
            nn.Dropout(dropout),
        )
       
    def forward(self, x: torch.Tensor):
        weights = self.net(x)
        weights = F.softmax(weights, dim = -1)
        return weights

def train(model: MoE, component_models: list[TransformerHangman],criterion: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epochs: int) -> None:
    """ training loop for mixture of experts """
    for epoch in range(epochs):
        total = 0
        for x, y in train_loader:
            #x,y = x.to(device),y.to(device)
            masks = batch_masks(length_of_batch_words(x))
            #masks.to(device)
            x = torch.mul(x.float(),masks)
            optimizer.zero_grad()
            weights = model(x)
            start = time.perf_counter()
            logits = models_outputs(component_models, x).to(device)
            print(time.perf_counter()-start)
            weights = weights[:,None,None,:].expand(-1,20,-1,-1)
            outputs = torch.matmul(weights,logits).squeeze()
            outputs = outputs.permute(0,2,1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total+=loss.item() #cumulative loss
        if epoch % 25 == 0:
            print('Epoch:', epoch, 'Loss:', total)

def train_and_save_MoE(model_files: list[str], dataset_file: str, num_neurons: int, max_iter: int, save_file: str) -> None:
    """ trains and saves the MoE """
    model = MoE(num_neurons, len(model_files))
    model = model.to(device)
    component_models = load_models(model_files)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=27)
    train_loader = DataLoader(load_dataset(dataset_file),batch_size,shuffle = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model,component_models,criterion,train_loader,optimizer,max_iter)
    with open(os.path.join('models', save_file), "wb") as file: pickle.dump(model, file)
    return

if __name__ == '__main__':
    """ trains the MoE model """
    model_files = ['prefix_model.pkl', 'suffix_model.pkl', '3gram_model.pkl', '4gram_model.pkl', '5gram_model.pkl', '6gram_model.pkl', '7gram_model.pkl', '8gram_model.pkl']
    train_and_save_MoE(model_files,'padded_words.pkl',num_neurons,max_iterations,'MoE.pkl')
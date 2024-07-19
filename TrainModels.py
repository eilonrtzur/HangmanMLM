import pickle
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import check_random_state
from torch.autograd import Variable
from itertools import product
import math
from CreateData import TrainData
torch.set_printoptions(sci_mode=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64
n_head = 8
n_layer = 2
dropout=0.2
learning_rate = 0.002
batch_size = 128
vocab_size = 27
max_iterations = 100

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ expanding linear layer and a non-linearity followed by shrinking linear layer and dropout """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        
        return x

class PositionalEncoding(nn.Module):
    """ Fixed positional encoding """
    def __init__(self, max_length: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(1000) / n_embd))
        pe = torch.zeros(max_length, n_embd)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
    
class TransformerHangman(nn.Module):
    """ combine attention blocks, positional encoding and a final linear layer """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = PositionalEncoding(8)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor):
        # B, T = idx.shape
        idx = idx.type(torch.LongTensor)
        idx = idx.to(device) 
        tok_emb = self.token_embedding_table(idx) # (B,T,n_emb)
        x = self.positional_encoding(tok_emb)
        x = self.blocks(x) # (B,T,n_emb)
        x = self.ln_f(x) 
        logits = self.lm_head(x) # (B,T,vocab_size)
        return logits

def set_of_masks(dimension: int) -> torch.Tensor:
    """ generate a set of masking vectors to censor unknown letters in all possible positions """
    masks = torch.Tensor(list(product([0, 1],repeat=dimension)))
    return masks[:-1,:]

def random_masks(masks: torch.Tensor, num_masks: int) -> torch.Tensor:
    """ Creates a sequence of num_masks random masks in bijection with a sequence of words """
    block_size = masks[0].size()[0]
    mask_seq = torch.zeros(num_masks,block_size)
    rand_ints = torch.randint(masks.size()[0], (num_masks,))
    for i in range(num_masks):
        mask_seq[i,:] = masks[rand_ints[i]]
    return mask_seq

def train(model: TransformerHangman, criterion: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epochs: int, block_size: int) -> None:
    """ training loop with printed error every 10 epochs """
    masks = set_of_masks(block_size)
    masks = masks.to(device)
    for epoch in range(epochs):
        total = 0
        for x, y in train_loader:
            x,y = x.to(device),y.to(device)
            batch_mask = random_masks(masks,len(x))
            batch_mask = batch_mask.to(device)
            x = torch.mul(x.float(),batch_mask)
            optimizer.zero_grad()
            outputs = model(x)
            outputs = outputs.permute(0,2,1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total+=loss.item() #cumulative loss
        if epoch % 10 == 0:
            print('Epoch:', epoch, 'Loss:', total)

def load_dataset(file_name: str) -> TrainData:
    """ function loading a dataset from the file name """
    file = open(os.path.join('datasets',file_name), "rb")
    dataset = pickle.load(file)
    return dataset

def train_and_save_model(save_file: str, criterion: nn.Module, train_loader: DataLoader, max_iterations: int, block_size: int) -> None:
    """ trains and saves a model """
    model = TransformerHangman()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model,criterion,train_loader,optimizer,max_iterations, block_size)
    with open(os.path.join('models', save_file), "wb") as file: pickle.dump(model, file)
    return

def train_all_models(datasets: list[TrainData], save_files: list[str]) -> None:
    """ trains and saves all models one by one """
    criterion = torch.nn.CrossEntropyLoss()  
    for i in range(len(datasets)):
        dataset = load_dataset(datasets[i])
        train_loader = DataLoader(dataset,batch_size,shuffle = True)
        train_and_save_model(save_files[i],criterion,train_loader,max_iterations, dataset.__block_size__())
    return

if __name__ == '__main__':
    datasets = ['prefix.pkl','suffix.pkl','3gram.pkl','4gram.pkl','5gram.pkl','6gram.pkl','7gram.pkl','8gram.pkl']
    save_files = [(datasets[i].split('.')[0] + '_model.pkl') for i in range(len(datasets))]
    train_all_models(datasets, save_files)

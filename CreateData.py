import string
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from itertools import product, combinations

# Import Data
full_dictionary_location = "wiki-100k.txt"

# Create Dataset
def build_dictionary(dictionary_file_location):
    text_file = open(dictionary_file_location,"r")
    full_dictionary = text_file.read().lower().split()
    text_file.close()
    return full_dictionary

# Only want to consider words of length at least 5
def filter_dictionary(dictionary,length):
    new_dictionary = []
    for word in dictionary:
        if len(word) >= length:
            new_dictionary.append(word)
    return new_dictionary

# creates dictionary and saves it
def create_and_save_dictionary(dictionary_file_location,length, save_location):
    dictionary = filter_dictionary(build_dictionary(dictionary_file_location),length)
    with open(save_location, "wb") as file: pickle.dump(dictionary, file)
    return dictionary

# Filter for n-grams of length n which will be trained separately, mode can be specified to prefix or suffix
def length_filter(dictionary,word_length = None, mode = None):
    new_dictionary = []
    if mode == 'prefix':
        for word in dictionary:
            new_dictionary.append(word[:3])
    elif mode == 'suffix':
        for word in dictionary:
            new_dictionary.append(word[-3:])
    else:
        for word in dictionary:
            if len(word)>= word_length:
                for i in range(len(word)-word_length):
                    new_dictionary.append(word[i:i+word_length])
    return new_dictionary

# defines our function which changes strings to a list of integers
def encode_function():
    chars = [letter for letter in string.ascii_lowercase]
    chars.insert(0,'.')
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    return encode

# encodes dictionary using encoder
def encoding_data(words):
    data = [encode(word) for word in words]
    data = torch.Tensor(data)
    return data

# Generate masks to censor unkown letters in all possible positions
# For example, if word status is _ _ l _ o, the mask will censor positions 0, 1, and 3
def set_of_masks(dimension):
    censors = torch.Tensor(list(product([0, 1],repeat=dimension)))
    masks = torch.zeros((2**dimension,dimension,dimension))
    for i in range(len(censors)):
        masks[i,:,:] = torch.diag(censors[i,:])
    return masks[:-1,:,:]

# Defines dataset
class Train_Data(Dataset): 
    # Constructor
    def __init__(self, block_size,words, mode = None):
        data = encoding_data(length_filter(words,block_size,mode))
        N_s=len(data) - block_size
        self.x = torch.zeros((N_s, block_size))
        self.y = torch.zeros((N_s, block_size))
        for i in range(N_s):
            self.x[i,:] = data[i]
            self.y[i,:] = data[i] 
        self.x = Variable(self.x.type(torch.long))
        self.y = Variable(self.y.type(torch.long))
        self.len = N_s
        self.block_size = block_size

    # Getter
    def __getitem__(self, index):    
        return self.x[index],self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len

    # Get block_size
    def __block_size__(self):
        return self.block_size

# Creates and saves the dataset 
def create_and_save_dataset(block_size, words, save_location, mode = None):
    dataset = Train_Data(block_size,words,mode)
    with open(os.path.join('datasets', save_location), "wb") as file: pickle.dump(dataset, file)
    return

# Creates datasets for prefix, suffix, and n-grams up to length 8
def create_datasets():
    create_and_save_dataset(3,words,'prefix.pkl','prefix')
    create_and_save_dataset(3,words,'suffix.pkl','suffix')
    create_and_save_dataset(3,words,'3gram.pkl')
    create_and_save_dataset(4,words,'4gram.pkl')
    create_and_save_dataset(5,words,'5gram.pkl')
    create_and_save_dataset(6,words,'6gram.pkl')
    create_and_save_dataset(7,words,'7gram.pkl')
    create_and_save_dataset(8,words,'8gram.pkl')
    return

# executes if this file is main file
if __name__ == '__main__':
    words = create_and_save_dictionary(full_dictionary_location,5,'allwords.pkl')
    encode = encode_function()
    create_datasets()
    print('All done.')


    


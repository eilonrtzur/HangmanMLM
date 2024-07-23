import string
import os
import pickle
import torch
from typing import Callable
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np

full_dictionary_location = "wiki-100k.txt"

def build_dictionary(dictionary_file_location: str) -> list[str]:
    """
    Takes in the desired dictionary file location and creates a list containing the individual words
    """
    text_file = open(dictionary_file_location,"r")
    full_dictionary = text_file.read().lower().split()
    text_file.close()
    return full_dictionary

def filter_dictionary(dictionary: list[str], min_size: int) -> list[str]:
    """
    Takes a list of words and removes words of length less than min_size
    """
    new_dictionary = []
    for word in dictionary:
        if len(word) >= min_size:
            new_dictionary.append(word)
    return new_dictionary

def create_and_save_dictionary(dictionary_file_location: str, min_size: int, save_location: str) -> list[str]:
    """
    Creates and saves a dictionary of words of length at least min_size
    """
    dictionary = filter_dictionary(build_dictionary(dictionary_file_location),min_size)
    with open(save_location, "wb") as file: pickle.dump(dictionary, file)
    return dictionary

def length_filter(dictionary: list[str], ngram_length: int = None, mode: str = None) -> list[str]:
    """
    If mode is prefix (suffix), takes a dictionary and creates a new list of all prefixes (suffixes) with repetitions.
    If mode is not specified, creates a new list of all substrings of length ngram_length found in dictionary, with repetitions. 
    """
    if mode == None and ngram_length == None:
        print('Error: Must specify word_length if mode is not specified')
    new_dictionary = []
    if mode == 'prefix':
        for word in dictionary:
            new_dictionary.append(word[:3])
    elif mode == 'suffix':
        for word in dictionary:
            new_dictionary.append(word[-3:])
    elif mode == 'pad':
        for word in dictionary:
            if len(word) == ngram_length:
                new_dictionary.append(word)
            else:
                for i in range(ngram_length - len(word)):
                    word += '_'
                new_dictionary.append(word)
        new_dictionary = np.random.choice(new_dictionary,5000) #MoE runs slow so I wanted smaller sample

    else:
        for word in dictionary:
            if len(word)>= ngram_length:
                for i in range(len(word)-ngram_length):
                    new_dictionary.append(word[i:i+ngram_length])
    return new_dictionary

def encode_function() -> Callable[[str], list[int]]:
    """
    Defines an encode function which translates a string into a list of integers
    """
    chars = [letter for letter in string.ascii_lowercase]
    chars.insert(0,'.')
    chars.insert(27,'_')
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    return encode

def encoding_data(words: list[str]) -> torch.Tensor:
    """
    Encodes list of words as array of integer arrays
    """
    data = [encode(word) for word in words]
    data = torch.Tensor(data)
    return data

class TrainData(Dataset): 
    """ Defines dataset class TrainData. Requires block_size and dictionary of words """

    # Constructor
    def __init__(self, block_size: int, words: list[str],mode: str = None):
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
    def __getitem__(self, index: int):    
        return self.x[index],self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len

    # Get block_size
    def __block_size__(self):
        return self.block_size

def create_and_save_dataset(block_size: int, words: list[str], save_location: str, mode: str = None) -> None:
    """
    Creates a saves a dataset given block size, dictionary of words, and save location
    """
    dataset = TrainData(block_size,words,mode)
    with open(os.path.join('datasets', save_location), "wb") as file: pickle.dump(dataset, file)
    return

def create_datasets() -> None:
    """
    Creates and saves datasets for each of the different types
    """
    create_and_save_dataset(3,words,'prefix.pkl','prefix')
    create_and_save_dataset(3,words,'suffix.pkl','suffix')
    create_and_save_dataset(3,words,'3gram.pkl')
    create_and_save_dataset(4,words,'4gram.pkl')
    create_and_save_dataset(5,words,'5gram.pkl')
    create_and_save_dataset(6,words,'6gram.pkl')
    create_and_save_dataset(7,words,'7gram.pkl')
    create_and_save_dataset(8,words,'8gram.pkl')
    create_and_save_dataset(20,words,'padded_words.pkl','pad')
    return

# executes if this file is main file
if __name__ == '__main__':
    words = create_and_save_dictionary(full_dictionary_location,5,'allwords.pkl')
    encode = encode_function()
    create_datasets()
    print('All done.')


    


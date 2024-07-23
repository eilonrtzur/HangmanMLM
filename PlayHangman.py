import string
from string import ascii_letters
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from CreateData import TrainData, encode_function, filter_dictionary, build_dictionary
from TrainModels import Head, MultiHeadAttention, Block, FeedFoward, PositionalEncoding, TransformerHangman
from MixtureofExperts import MoE, load_models
np.set_printoptions(precision=2,suppress=True)

def decode_function() -> Callable[[list[int]], str]:
    """ Defines a decode function which turns an array of numbers into a string of characters """
    chars = [letter for letter in string.ascii_lowercase]
    chars.insert(0,'.')
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    return decode

def load_MoE(save_file: str) -> MoE:
    """ loads MoE model """
    file = open(os.path.join('models', save_file),"rb")
    model = pickle.load(file)
    return model

def model_output(model: TransformerHangman, status: str) -> torch.Tensor:
    """ evaluates a model on a given input string representing the status of the hangman game and converts to probabilities """
    m = nn.Softmax(dim=2)
    encode = encode_function()
    word = torch.Tensor(encode(status))
    word = word[None,:]
    output = model(word)
    output = m(output)
    output = torch.sum(output,1)
    output = torch.squeeze(output)
    result = output.detach()
    result[0]=0
    return result

def moe_output(model: MoE, status: str) -> torch.Tensor:
    """ evaluates MoE model and outputs weights """
    encode = encode_function()
    for i in range(20-len(status)):
        status += '_'
    padded_word = torch.Tensor(encode(status))
    padded_word = padded_word[None,:]
    output = model(padded_word)
    output = output.squeeze().detach()
    return output
        
def remove_guessed(model_output: torch.Tensor, guessed_letters: list[int]) -> torch.Tensor:
    # sets probabilities of guessed letters to 0 and renormalizes
    for num in guessed_letters:
        model_output[num] = 0

    return model_output

def generate_prediction(models: list[TransformerHangman], moe_model: MoE, guessed_letters: list[str], status: str) -> str:
    """ generate next letter prediction based on current status and previously guessed letters """
    encode = encode_function()
    decode = decode_function()
    guessed_letters = [encode(letter)[0] for letter in guessed_letters] # encode guessed letters to use later
    length = len(status)
    logits = [] # create a vector of logit outputs from individual component which can then be multiplied by weights
    logits.append(remove_guessed(model_output(models[0].eval(), status[:3]),guessed_letters))
    logits.append(remove_guessed(model_output(models[1].eval(), status[-3:]),guessed_letters))
    for j in range(2,min(len(models),length)): # create a loop for n-gram models
        logits.append(remove_guessed(torch.sum(torch.stack([model_output(models[j].eval(),status[i:i+j+1]) for i in range(length-j)]),dim=0),guessed_letters)/(length-j))  
    logits = torch.stack(logits)
    weights = moe_output(moe_model.eval(),status)[:logits.size(0)]
    print(status, weights)
    # weights = torch.ones(logits.size(0))
    result = torch.matmul(weights,logits)
    result = result.numpy()
    prediction = (-result).argsort()
    prediction = prediction.tolist()
    return decode([prediction[0]])
  
def update_word(status: str, word: str, letter: str) -> str:
    """ updates the status of the word with guessed letter"""
    new_status = ''.join([status[i] if word[i]!= letter else letter for i in range(len(status))])
    return new_status

def play_hangman(word: str, mode: str = None) -> int:
    """ 
    simulates an entire game of hangman for chosen word and component weights
    if mode is 'print', it will print out responses for each round of letter guessing in game
    returns 1 for a win and 0 for a loss 
    """
    guessed_letters = []
    incorrect_guesses = 0
    status = ''
    for i in range(len(word)):
        status += '.'

    while status != word and incorrect_guesses<7:
        letter = generate_prediction(models,moe_model,guessed_letters,status)
        guessed_letters.append(letter)
        if letter in word:
            status = update_word(status,word,letter)
            if mode == 'print':
                print('The letter', letter,'is in the word!')
                if status != word:
                    print('The state is now ',status)
            continue
        else:
            incorrect_guesses += 1
            if mode == 'print':
                if incorrect_guesses<7:
                    print('There is no', letter,'in the word. You have',6-incorrect_guesses, 'mistakes left.')
                else: 
                    print('There is no', letter,'in the word.')

    
    if status == word:
        if mode == 'print':
            print('You win! The word was ', word)
        return 1
    else:
        if mode == 'print':
            print('You lose! The word was', word)
        return 0

def win_rate(sample: list[str]) -> float:
    """ returns percentage of games won for a given sample of words """
    games_won = 0
    games_played = len(sample)
    for word in sample:
        if play_hangman(word)==1:
            games_won += 1
    return games_won/games_played

def create_sample(dictionary_file: str, sample_size: int, seed: int = None) -> list[str]:
    """ creates a random sample of words of a given size from a given dictionary  """
    file = open(dictionary_file, "rb")
    words = pickle.load(file)
    if sample_size > len(words):
        print('Error: Sample size larger than dictionary.')
        return
    # random generator for testing purposes
    if seed != None:
        rng = np.random.default_rng(seed)
        sample = rng.choice(words,sample_size)
    else:
        sample = np.random.choice(words,sample_size)
    return sample

if __name__ == '__main__':
    """
    Running this script will prompt the user to enter either "game" or "win-rate". If "game" is entered, then it will ask for a word. 
    If "win-rate" is entered, it will ask for a sample size. 
    """

    model_files = ['prefix_model.pkl', 'suffix_model.pkl', '3gram_model.pkl', '4gram_model.pkl', '5gram_model.pkl', '6gram_model.pkl', '7gram_model.pkl', '8gram_model.pkl']
    models = load_models(model_files)
    moe_model = load_MoE('MoE.pkl')
    response = input('Choose game or win-rate: ')
    if response == 'game':
        word = input('Choose a word or type random please for a random word: ')
        word = word.lower()
        if word != 'random please':
            if set(word).difference(ascii_letters):
                print('Your word must only include english letters, no special characters are allowed.')
            else:    
                play_hangman(word,'print')
        else:
            play_hangman(create_sample('allwords.pkl',1)[0],'print')
    elif response == 'win-rate':
        sample_size = int(input('How big a sample would you like? '))
        if sample_size < 1:
            print('You entered an invalid sample size.')
        else:
            print('Your win rate is ',win_rate(create_sample('allwords.pkl',sample_size,1)))
    else: 
        print('That is not an option. Please choose type game or win-rate next time.')

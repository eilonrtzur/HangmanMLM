import string
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from CreateData import Train_Data, encode_function
from TrainModels import Head, MultiHeadAttention, Block, FeedFoward, PositionalEncoding, TransformerHangman

np.set_printoptions(precision=2,suppress=True)

# defines our decoding function, turning array of numbers back into string
def decode_function():
    chars = [letter for letter in string.ascii_lowercase]
    chars.insert(0,'.')
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    return decode

# loads models
def load_models(save_files):
    models = []
    for save_file in save_files:
        file = open(os.path.join('models', save_file),"rb")
        models.append(pickle.load(file))
    return models

# run model component and output logits
def model_output(model, status):
    m = nn.Softmax(dim=2)
    encode = encode_function()
    word = torch.Tensor(encode(status))
    word = word[None,:]
    output = model(word)
    output = m(output)
    output = torch.sum(output,1)
    output = torch.squeeze(output)
    result = output.detach().numpy() 
    result[0]=0
    return result

# sets probabilities of already guessed letters to 0 and re-normalizes and squares to emphasize confidence
def remove_guessed(model_output,guessed_letters):
    for num in guessed_letters:
        model_output[num] = 0
    if sum(model_output) != 0:
        model_output = np.power(model_output/sum(model_output),2) # we raise to power to emphasize confidence in a prediction
    return model_output

# Generate predictions based off current status and previously guessed letters
def generate_prediction(models,guessed_letters,status, weights):
    encode = encode_function()
    decode = decode_function()
    guessed_letters = [encode(letter)[0] for letter in guessed_letters] # encode guessed letters to use later
    length = len(status)
    logits = [] # create a vector of logit outputs from individual component which can then be multiplied by weights
    logits.append(remove_guessed(model_output(models[0].eval(), status[:3]),guessed_letters))
    logits.append(remove_guessed(model_output(models[1].eval(), status[-3:]),guessed_letters))
    for j in range(2,min(len(models),length)): # create a loop for n-gram models
        logits.append(sum(remove_guessed(model_output(models[j].eval(),status[i:i+j+1]),guessed_letters) for i in range(length-j))/(length-j))
    result = sum(weights[i]*logits[i] for i in range(len(logits)))
    prediction = (-result).argsort()
    prediction = prediction.tolist()

    return decode([prediction[0]])

# we update the status of the word if our guessed letter is found in the word   
def update_word(status,word,letter):
    new_status = ''.join([status[i] if word[i]!= letter else letter for i in range(len(status))])
    return new_status

# this function simulates the entire game of hangman for the chosen word with component weights
# if mode is 'print', it will play the game and print statements for each guess
def PlayHangman(word, weight = None, mode = None):
    if weight == None:
        weight = [1 for i in range(len(word))]
    else:
        weight = weight[:len(word)]
    guessed_letters = []
    incorrect_guesses = 0
    global component_success

    status = ''
    for i in range(len(word)):
        status += '.'

    while status != word and incorrect_guesses<7:
        if mode == 'weight_train':
            (letter, components_array) = generate_prediction(models,guessed_letters,status, weight)
            for i in range(len(components_array)):
                if components_array[i] in word:
                    component_success[i]+=1
                
        else:
            letter = generate_prediction(models,guessed_letters,status, weight)
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

# Computes the percentage of games won over a given sample of words and a choice of component weights
def WinRate(sample,weight = None):
    games_won = 0
    games_played = len(sample)
    for word in sample:
        if PlayHangman(word, weight)==1:
            games_won += 1
    return games_won/games_played

# creates a random sample of words from dictionary file, is used to test win-rate. 
def create_sample(dictionary_file,sample_size):
    file = open(dictionary_file, "rb")
    words = pickle.load(file)
    if sample_size > len(words):
        print('Error: Sample size larger than dictionary.')
        return
    sample = np.random.choice(words,sample_size)
    return sample

# Running this script will prompt the user to enter either "game" or "win-rate". If "game" is entered, then it will ask for a word. If 
# "win-rate" is entered, it will ask for a sample size. 
if __name__ == '__main__':
    model_files = ['prefix_model.pkl', 'suffix_model.pkl', '3gram_model.pkl', '4gram_model.pkl', '5gram_model.pkl', '6gram_model.pkl', '7gram_model.pkl', '8gram_model.pkl']
    models = load_models(model_files)
    response = input('Choose game or win-rate: ')
    if response == 'game':
        word = input('Choose a word: ')
        PlayHangman(word,None,'print')
    elif response == 'win-rate':
        sample_size = int(input('How big a sample would you like? '))
        print('Your win rate is ',WinRate(create_sample('allwords.pkl',sample_size)))
    else: 
        print('That is not an option. Please choose type game or win-rate next time.')

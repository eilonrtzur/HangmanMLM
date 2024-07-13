import string
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from CreateData import Train_Data, encode_function
from TrainModels import Head, MultiHeadAttention, Block, FeedFoward, PositionalEncoding, TransformerHangman


torch.set_printoptions(sci_mode=False)
np.set_printoptions(precision=2,suppress=True)
n_embd = 32
n_head = 4
head_size = n_embd // n_head
dropout= 0.2
n_layer = 4
learning_rate = 0.002
batch_size = 128

def decode_function():
    chars = [letter for letter in string.ascii_lowercase]
    chars.insert(0,'.')
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    return decode

def load_models(save_files):
    models = []
    for save_file in save_files:
        file = open(save_file,"rb")
        file.eval()
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

def remove_guessed(model_output,guessed_letters):
    for num in guessed_letters:
        model_output[num] = 0
    if sum(model_output) != 0:
        model_output = np.power(model_output/sum(model_output),2) # we raise to power to emphasize confidence in a prediction
    return model_output

# Generate predictions based off current status and previously guessed letters
# weight should be a vector of length 5 representing [prefix, suffix, trigram, fourgram, specific length]
def generate_prediction(models,guessed_letters,status, weight):
    encode = encode_function()
    decode = decode_function()
    guessed_letters = [encode(letter)[0] for letter in guessed_letters] # encode guessed letters to use later
    length = len(status)
    logits = [] # create a vector of logit outputs from individual component which can then be multiplied by weights
    logits.append(remove_guessed(model_output(models[0], status[:3]),guessed_letters))
    logits.append(remove_guessed(model_output(models[1], status[-3:]),guessed_letters))
    for j in range(2,len(models)): # create a loop for n-gram models
        logits.append(sum(remove_guessed(model_output(models[j],status[i:i+j+1]),guessed_letters) for i in range(length-(j+1)))/(length-(j+1)))
    result = sum(weight[i]*logits[i] for i in range(len(logits)))
    prediction = (-result).argsort()
    prediction = prediction.tolist()

    return decode([prediction[0]])

# we update the status of the word if our guessed letter is found in the word   
def update_word(status,word,letter):
    new_status = ''.join([status[i] if word[i]!= letter else letter for i in range(len(status))])
    return new_status

# this function simulates the entire game of hangman for the chosen word with component weights
# if mode is not None, it will play the game and print statements for each guess
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
            letter = generate_prediction(models,guessed_letters,status, weight, mode)
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

def create_sample(dictionary_file,sample_size):
    file = open(dictionary_file, "rb")
    words = pickle.load(file)
    if sample_size < len(words):
        print('Error: Sample size smaller than dictionary.')
        return
    sample = np.random.choice(words,sample_size)
    return sample


# # Naive random training of weights
# def train_weights(sample_size,jump_size, iterations, w_0):
    
#     movements = abs(np.random.normal(0,jump_size,iterations)) # the size and direction of the adjustments 
#     current_weights = w_0
    
#     for i in range(iterations):
#         sample = np.random.choice(words[:5000],sample_size)
#         global component_success
#         component_success = np.zeros(len(initial_weights))
#         accuracy = SuccessRate(sample,current_weights, 'weight_train')
#         print(component_success)
#         component_success = component_success / sum(component_success)
#         current_weights = np.add(current_weights, movements[i]*component_success) # implement adjustments
#         current_weights = current_weights/sum(current_weights)
#         print(i,accuracy,current_weights) # can print this to see how it's updating
#     return accuracy,current_weights


if __name__ == '__main__':
    model_files = ['prefix_model.pkl', 'suffix_model.pkl', '3gram_model.pkl', '4gram_model.pkl', '5gram_model.pkl', '6gram_model.pkl', '7gram_model.pkl', '8gram_model.pkl']
    models = load_models(model_files)
    response = input('Choose game or win-rate')
    if response == 'game':
        word = input('Choose a word!')
        PlayHangman(word,'print')
    elif response == 'win-rate':
        sample_size = input('How big a sample would you like?')
        print('Your win rate is ',WinRate(create_sample('allwords.pkl',sample_size)))

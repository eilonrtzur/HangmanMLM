# The Hangman Project

This is a masked language model constructed to predict letters in the game of Hangman. The purpose of this project was to get some experience building the model from scratch while applying it to a specific problem. A typical GPT-like model predicts/generates the next token given a sequence of tokens via an intuitive "look-ahead" mask. The Hangman situation differs in that we must predict a token (in our case, a letter) that may appear anywhere in the word where there is not already a known letter. Originally, we chose to tackle this by taking the set of all masks on a given sequence length and training our model on the set of masks applied to our corpus, representing all possible states found while playing Hangman. Since the set of masks double with every increment to our block size, the datasets were growing exponentially. Therefore, we instead elected to randomly choose a mask during training and apply it there. This can be interpreted as masking every letter at random with a 50% chance. We trained models for prefixes (first three letters), suffixes (last three letters), and n-grams of lengths 3-8, then combined the output of these models when predicting the next guessed letter in Hangman. 
This README provides an overview of the project structure and instructions for running the application. Feel free to train your own models, tweak parameters, change the corpus, and see how it performs! I've also included some barely trained models so you can try playing without having to train.

This method could also be extended to solve problems such as filling in missing text within a larger document, recovering parts of images, and anything else involving incomplete data.

## Project Structure

The project consists of the following files and components:

### Files

- **CreateData.py**: This file is used to create the datasets which will be used to train the models. 
  
- **TrainModels.py**: This file builds the models and trains them. 

- **PlayHangman.py**: This file takes in the train models and allows you to fine-tune the combined model, play Hangman, and check it's success rate over a large sample of words.

- **datasets**: This folder contains files with saved datasets which can be used for training. 

- **models**: This folder contains models with slightly (100 iterations) trained models which can be used to play hangman and test accuracy. 

- **words_250000_train.txt**: File containing 250,000 common English words (and some names). It's been filtered to only contain words of length at least 5, so probably fewer words now.  

- **allwords.pkl**: A file containing a list of the words in the corpus of length at least 5, since any shorter words should be illegal in Hangman. 




## How to Use

Coming soon.

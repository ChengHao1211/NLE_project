# NLP_PROJECT

## Overview
This jupyter notebook is for question similarity based on BERT (Bidirectional Encoder Representations from Transformers). The model is trained to determine whether two input questions are similar or not. And you can get that how much the percentage for the similarity.

## Environment of the code
You can run the code in the server witch has Anaconda environment and GPU for machine learning.
Here is the instruction of how to set up the Anaconda environment and GPU
https://medium.datadriveninvestor.com/installing-tensorflow-gpu-using-anaconda-on-windows-ac23b66d05f1


## The way to run the code
All you can do for running the code is just click on running in the top bar. But the most important thing is the path of the data file. You have to modify the data file path in the code "train_df = pd.read_csv("Datafile_path")" and "test_df = pd.read_csv("Datafile_path")". All you need to do is jsut modify the Datafile_path to the path of the file. You can also put the zip file in the path. It will work.

## The libaries you may install
Here is some of the libraries you may install to run this code.
numpy (pip install numpy)
pandas (pip install pandas)
torch (pip install torch)
seaborn (pip install seaborn)
matplotlib (pip install matplotlib)
tqdm (pip install tqdm)
transformers (pip install transformers)


## Data analysis
The code performs exploratory data analysis on the training data, including visualizations of question lengths, shared word distributions, and a pie chart showing the distribution of similar and different question pairs. All of the graphs you can see in the code.

## The Model for training
The BERT model used in this code is based on the 'bert-base-uncased' version. The model is fine-tuned for question similarity using PyTorch and the Hugging Face transformers library. 

## Data Preparation
The data is split into training and validation sets for model training. Then code tokenizes and encode the input questions using encode_plus function. After geting the encode data I used torch.tensor to get the data for the trainning model as an input data. And the data will be the "Id of the questions", "mask", "token id" and "target".  
The input data for the model will look like this.
Input IDs: tensor([[  101,  1045,  2215,  ...,     0,     0,     0],
        [  101,  2339,  2003,  ...,     0,     0,     0],
        [  101,  2129,  2079,  ...,     0,     0,     0],
        ...,
        [  101,  2129,  2079,  ...,     0,     0,     0],
        [  101,  2129,  2097,  ...,     0,     0,     0],
        [  101, 22817, 14181,  ...,     0,     0,     0]])
Attention Mask: tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]])
Token Type IDs: tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]])
Targets: tensor([1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,
        0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
        1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0])


## Model Training
The model is trained using the training data and the validation set. Training progress is logged, and the model with the lowest validation perplexity is saved in the models directory.
For the trainning I use Learning Rate (lr) = 3e-5, Batch Size  = 256 and Number of Epochs = 5 to train the model. The output of the trainning will look like this 
Epoch:  1
| Iter 100 | Avg Train Loss 0.36829567819833753 | Dev Perplexity 1.4120122640005845
| Iter 200 | Avg Train Loss 0.3381352695822716 | Dev Perplexity 1.3721972345022608

And the end of my trainning I run into training Loss 0.07... Which is not bad.
| Iter 5900 | Avg Train Loss 0.06461081946268678 | Dev Perplexity 1.389561617659267
| Iter 6000 | Avg Train Loss 0.06891279976814985 | Dev Perplexity 1.3900370267414317
| Iter 6100 | Avg Train Loss 0.07039290873333812 | Dev Perplexity 1.3585926732200044
You can see the result look fine that the Loss decrease to 0.070.

## Testing
The trained model is then used to predict the questions similarity on the test set, and the results are saved in the result.csv file. The function here will give you the percentage of the prediction between question1 and question2 in the test dataset.

## Results 
In the code I have printed out head 100 results of the prediction and two types of the visualized graph.
Scatter Plot: Shows the relationship between test IDs and predicted question similarity.
Bar Plot: Illustrates the distribution of question similarity predictions.

## What I learned from this project
How to use BERT model properly to train the data.
How to analyze the data and make a graph to easily understand.
The hyperparameters affect the training a lot.

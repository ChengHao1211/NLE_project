
# Quora Question Pairs

## Environment setup

#### 1.Install Anaconda by using this link - https://www.anaconda.com/download
#### 2.Launch Jupiter notebook
#### 3.Download NLP_FINAL_PROJECT.ipynb file and train.csv file from the ljampala folder
#### 4.Open this NLP_FINAL_PROJECT.ipynb file in jupiter notebook
#### 5.Copy the train.csv file path to the corresponding dataframe like this df = pd.read_csv("train.csv", encoding='ISO-8859-1')
#### 6.Run each cell or run all functions from the jupiter notebook
#### 7.Install the required libraries using pip install either in jupiter notebook or the Command Prompt.
Example pip install numpy, pip install pandas etc
### Problem Statement :
The problem statement involves creating a system to identify duplicate questions on the Quora platform. Duplicate identification is a common challenge in various online platforms, and in this context, the goal is to develop an algorithm  that can efficiently recognize whether a given question is a duplicate of one that has already been asked.

### Abstract
Quora is a place to gain and share knowledge about anything. It’s a website to ask questions and connect with people who contribute unique insights and quality answers. Over 100 million people visit Quora every month, it's no surprise that many people ask similarly worded questions. But so many questions cause a lot of same questions with different word or different way to ask. These multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. So, we try using the NLP technique to identify duplicate questions that can provide a better experience to active seekers and writers and offer more value to both groups in the long term.


### Proposed Solution Steps:

### Training dataset:
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. Total we have 404290 entries. Splitted data into train and test with 80% and 20%.
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/156c5562-77b3-4e60-bdeb-fcff7f8b0086)

### Analysis of the dataset
Any common words between the questions
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/13089ff2-0894-4768-92eb-67a25b74fc68)

#### Word sharing between the questions
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/5b1c8a75-f65d-48ce-94af-752941d3013a)

#### Data Preprocessing:
Clean and preprocess the question text data. This involves tasks like removing stop words, handling punctuation, and converting text to lowercase.
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/69569006-bca9-49e7-a172-0c65a65f00bd)

#### Splitting the dataset Example Image
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/c823e264-9597-4219-9ecf-48b288aa43ec)

### Model Selection:

RandomForestClassifier model using 5-fold cross-validation.
First, the collected dataset was divided into two parts: training data (80%) and test data (20%). While the training data are then used for model optimization and development, test data are kept separately to avoid any data leakage. To train deep learning models, 20% of total training data are used to create a validation set, and the rest of the data are used for model development. The validation set plays a role in finding the optimal models. To train the machine learning model, the training data are used without creating an independent validation set. A 5-fold cross-validation is applied to the training data to compute the average performance corresponding to specific sets of parameters to find the best hyperparameters. The machine learning models are then retrained with training data and hyperparameters. 

![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/d5753e27-7042-485d-a586-cd5a3fd0acd6)

## Results

### Evaluation:
Confusion Matrix:
Evaluate the model is done by the Metrics such as precision, recall, and F1 score can be used to measure the model's effectiveness.
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/6654edb1-87b5-4348-b918-cbdf44f20acb)

### Confusion Matrix:
#### Precision, Recall, and F1-Score:
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/3080328f-a530-4b11-9196-14e41adc5827)

### Receiver Operating Characteristic (ROC) Curve and Area Under the Curve (AUC):
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/e9fbdbf9-a0cd-461f-a3fb-5bd264d348e9)

**References:**
https://www.nature.com/articles/s41598-022-24979-9
https://peerj.com/articles/cs-1570/
https://www.kaggle.com/competitions/quora-question-pairs/code
https://github.com/campusx-official/quora-question-pairs/blob/main/bow-with-basic-features.ipynb
https://github.com/campusx-official/quora-question-pairs/blob/main/bow-with-basic-features.ipynb
https://www.javatpoint.com/machine-learning-random-forest-algorithm



# Quora Question Pairs

## Environment setup

#### 1.Install Anaconda by using this link - https://www.anaconda.com/download
#### 2.Launch Jupyter notebook
#### 3.Download NLP_FINAL_PROJECT.ipynb file and train.csv file from the ljampala folder
#### 4.Open this NLP_FINAL_PROJECT.ipynb file in jupyter notebook
#### 5.Copy the train.csv file path to the corresponding dataframe like this df = pd.read_csv("train.csv", encoding='ISO-8859-1')
#### 6.Run each cell or run all functions from the jupyter notebook
#### 7.Install the required libraries using pip install either in jupyter notebook or the Command Prompt.
pip install numpy, pip install pandas etc
### Problem Statement :
The problem statement involves creating a system to identify duplicate questions on the Quora platform. Duplicate identification is a common challenge in various online platforms, and in this context, the goal is to develop an algorithm  that can efficiently recognize whether a given question is a duplicate of one that has already been asked.

### Abstract
Quora is a place to gain and share knowledge about anything. It’s a website to ask questions and connect with people who contribute unique insights and quality answers. Over 100 million people visit Quora every month, it's no surprise that many people ask similarly worded questions. But so many questions cause a lot of same questions with different word or different way to ask. These multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. So, we try using the NLP technique to identify duplicate questions that can provide a better experience to active seekers and writers and offer more value to both groups in the long term.


### Proposed Solution Steps:

### Training dataset:
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. Total we have 404290 entries. Splitted data into train and test with 80% and 20%.

![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/8402d08d-71e4-4f27-baf2-5391c528cd73)

### Analysis of the dataset
#### Any common words between the questions
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/e94dc117-e328-408e-bf73-f6281014af45)

#### Word sharing between the questions
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/5b1c8a75-f65d-48ce-94af-752941d3013a)

#### Data Preprocessing:
Clean and preprocess the question text data. This involves tasks like removing stop words, handling punctuation, and converting text to lowercase.

![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/69569006-bca9-49e7-a172-0c65a65f00bd)

#### Splitting the dataset Example Image
![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/c823e264-9597-4219-9ecf-48b288aa43ec)

### Model Selection:

**RandomForestClassifier model:**
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
Random forests are a popular supervised machine learning algorithm. Random forests are for supervised machine learning, where there is a labeled target variable.
Random forests can be used for solving regression (numeric target variable) and classification (categorical target variable) problems.
Random forests are an ensemble method, meaning they combine predictions from other models.
Each of the smaller models in the random forest ensemble is a decision tree. 
In a random forest classification, multiple decision trees are created using different random subsets of the data and features. Each decision tree is like an expert, providing its opinion on how to classify the data. Predictions are made by calculating the prediction for each decision tree, then taking the most popular result. 

![image](https://github.com/ChengHao1211/NLE_project/assets/144284576/d5753e27-7042-485d-a586-cd5a3fd0acd6)

## Results
In the exploration of machine learning algorithms on the Quora dataset, several classifiers were employed, including RandomForestClassifier, SVC Classifier, KNeighborsClassifier, and XGBClassifier. Among these algorithms, RandomForestClassifier emerged as the most successful, delivering superior performance. After training the entire dataset, the RandomForestClassifier achieved an impressive 80% accuracy. This outcome underscores the effectiveness and suitability of the RandomForestClassifier for the specific characteristics of the Quora dataset, showcasing its robustness in handling the complexity of the data.
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


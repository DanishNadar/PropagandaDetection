import numpy as np
import pandas as pd
import re # Useful for searching text in a document
from nltk.corpus import stopwords # Stopwords will find words that do not add much necessary information, like articles, and removes them
from nltk.stem.porter import PorterStemmer # Stemming removes the prefix and/or the suffix of the word to return the base word
from sklearn.feature_extraction.text import TfidfVectorizer # Important to help convert speech into feature vectors (numbers), so that the computer can understand the inputs and produce accurate outputs
from sklearn.model_selection import train_test_split # Used to split a dataset into training data and testing data
from sklearn.linear_model import LogisticRegression # I will be using a logistic regression methodology due to the binary classification of true or false
from sklearn.metrics import accuracy_score
import nltk
from datetime import datetime
import joblib # A library that I used to save the model

#DATA PREPROCESSING

counter = 0
nltk.download('stopwords') # Downloading all of the unnecessary stopwords
dataset = pd.read_csv('/home/spiron/Documents/Development/COM 380 Propaganda Detection AI/train.csv') # Has 20800 articles and 5 features
dataset = dataset.fillna('') # Fills any missing values with a null string 

# Merging the news title, author name, and content
dataset['content'] = dataset['author'] + ' ' + dataset['title'] + ' ' + dataset['text']

x = dataset.drop(columns='label', axis=1) #'label' will be removed from the dataframe (axis = 1 means column and axis = 0 means row)
y = dataset['label'] # Since I removed 'label' from the dataframe for 'x,' I have created the variable 'y to contain all of that removed 'label' data

port_stem = PorterStemmer()
def stemming(content):
    # Removing all non-alphabetic characters and lowercasing all of the content
    global counter
    counter = counter + 1
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    with open('PropagandaDetection_StemmingLog.txt', 'a+') as file:
        file.write(f"\nWord ID: {counter}, Current Time: {current_time}, Content Length: {len(content)}")
    
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    
    stemmed_content = stemmed_content.split() # Splitting the content into individual words

    # Create a new list to store stemmed and filtered wordsv
    filtered_stemmed_content = []  

    # Iterate through each word and apply stemming
    for word in stemmed_content:
        stemmed_word = port_stem.stem(word)  # Apply stemming to the current word

        # Removing any unnecessary stopwords
        if not stemmed_word in stopwords.words('english'):  # Words will only be processed and stemmed if they are not in the stopwords library
            filtered_stemmed_content.append(stemmed_word)

    # Join all the processed words to form the filtered stemmed content string
    filtered_stemmed_content = ' '.join(filtered_stemmed_content)  
    
    return filtered_stemmed_content  # Return the filtered stemmed content

dataset['content'] = dataset['content'].apply(stemming) # This will take all of my content, including the title, author name, and text, and stem everything

# Separating the content and the label
X = dataset['content'].values
Y = dataset['label'].values

# Converting the text data into numerical data so the computer can interpret it
vectorizer = TfidfVectorizer() # Creating the vectorizer instance
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting the dataset to training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2) # We are splitting 80% of the data to be training data and 20% to be testing data

model = LogisticRegression()
model.fit(X_train, Y_train)

#accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy Score of the Training Data: " + str(training_data_accuracy))


#accuracy score on the testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy Score of the Testing Data: " + str(testing_data_accuracy))

# Save the trained model using joblib
model_filename = "propaganda_detection_modelWITHTEXT.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

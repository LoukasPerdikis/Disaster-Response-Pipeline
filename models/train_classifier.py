import sys

import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import pickle
import joblib

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

print('libraries installed successfully')


def load_data(database_filepath):
    '''
    Args:
    
    database_filepath: str, the filepath to the SQLite database
    
    Loads the data that was cleaned in a previous step and enables it for ML.
    
    Returns:
    
    X: pandas Dataframe, dataframe of features
    Y: pandas Dataframe, dataframe of targets
    category_names: list of strings, a collection of the target category names 
    '''
    # Load data from SQLite database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table('DisasterResponsePipeline', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    
    category_names = Y.columns.values
    
    print('Stage 1 passed: data imported successfully')
    
    return X, Y, category_names


def tokenize(text):
    '''
    Args:
    
    text: str, a long string with multiple words/ sentences.
    
    Tokenizes and cleans raw text data to include only relevant word tokens:
      - normalizes text by applying lower case and punctuation removal
      - tokenizes text using the nltk library
      - removes stopwords that would not be helpful for ML
      - applies lemmatization
      
    Returns:
    
    cleaned_text: list of strings, contains all tokenized, cleaned text words.
    '''
    # Normalize text
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in text if word not in stopwords.words('english')]
    
    # Lemmatization
    lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in words]
    cleaned_text = [WordNetLemmatizer().lemmatize(word, pos='v') for word in lemmatized_words]
    
    return cleaned_text


def build_model():
    '''
    Args: None
    
    ML pipeline which includes all vectorization and algorithm initialization, as well as gridsearch methodology to encourage retrieval of the best possible model configuration.
    
    Returns:
    
    model: scikit-learn model, a successfully trained ML model.
    '''
    # Machine learning pipeline initialized below
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier(n_jobs=-1)))
    ])
    print('Pipeline initialized')
    
    # parameters for Gridsearch. The parameter options are low due to program run time and for efficiency.
    parameters = {
        'clf__estimator__n_neighbors': [5, 10]
    }
    
    # Gridsearch
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2)
    print('Stage 2 passed: Gridsearch completed successfully')
    
    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Args:
    
    model: scikit-learn model, a successfully trained ML model.
    X_test: pandas DataFrame, feature variables included in testing set
    Y_test: pandas DataFrame, target variables included in testing set
    category_names: list of strings, a collection of the target category names
    
    Uses the trained model in the previous step to predict on unseen data and provide a classification report from scikit-learn.
    
    Returns: None
    '''
    # Use model to predict on unseen test data
    Y_pred = model.predict(X_test)
    print('Stage 4 passed: Unseen data predicted successfully')
    
    # Supply classification report with accuracy, precision, recall, and f1 scores
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print('Stage 5 passed: Classification report supplied successfully')


def save_model(model, model_filepath):
    '''
    Args:
    
    model: scikit-learn model, a successfully trained ML model.
    model_filepath: str, the model_filepath
    
    Saves the trained model for future use using joblb.
    
    Returns: None
    '''
    # Save model as pickle file
    joblib.dump(model, model_filepath)
    print('Stage 6 passed: Model saved successfully')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Stage 4 passed: Model trained successfully')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

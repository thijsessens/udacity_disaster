import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    '''this function loads data from sqlite database'''
    
    # connect to engine sqlalchemy
    engine = create_engine('sqlite:///'+database_filepath)
    
    # create df
    df = pd.read_sql_table('data', engine)
    
    # create X,Y and category_names
    X = df.iloc[:, 1]
    Y = df.iloc[:, 4:]
    category_names = list(df.iloc[:, 4:].columns)
    
    return X, Y, category_names


def tokenize(text):
    '''this function creates tokens from words'''
    
    # detect url 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    #initiate tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # loop through words
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''this function initiates the model as a pipeline'''
    
    # initiate pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''this function evaluates the model through classification_report'''
    
    # create predictions to from X_test
    Y_pred = model.predict(X_test)
    
    # get 1D array for classification_report input
    Y_test = Y_test.values
    
    # loop through all categories and print classification_report
    i = 0
    while i < len(category_names):
        print("below is the classification report for: "+category_names[i])
        print(classification_report(Y_test[i], Y_pred[i]))
        i += 1


def save_model(model, model_filepath):
    '''this function saves the model on pickle'''
    
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''this is the main function'''
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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

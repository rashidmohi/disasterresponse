import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    INPUT
    database_filepath - string [path and filename of the source db file]
    
    OUTPUT
    X - numpy ndarray containing the predictor dataset for modelling
    Y - numpy ndarray containing the target dataset for modelling
    cat_names - numpy ndarray containing the names of the target variable(s)
    

    This function performs following steps to produce output dataframe:
    1. Load the disaster messages data from the db file
    2. Splits and returns the predictor variables dataset, target variables dataset and names of
    target variables
    '''
    # load data from database
    db_file = 'sqlite:///'+database_filepath
    print(db_file)
    engine = create_engine(db_file)
    df = pd.read_sql("SELECT * FROM disaster_messages", engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    cat_names = df.iloc[:,4:].columns.values
    
    return X, Y, cat_names

def tokenize(text):
    '''
    INPUT
    text - string [raw text to be tokenized]
    
    OUTPUT
    tokens - list of cleansed and lematized tokens
    

    This function tokenize the given text for TFIDF vectorization
    '''
    # initialize lemmatizer and stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    INPUT
    None (uses the training dataset created earlier)
    
    OUTPUT
    model - ML model for predicting the disaster categories (multi) based on input text
    
    

    This function creates a model for predicting the response categories based on the message.
    Evaluated Random Forest and Multinomial Naive Bayes classifiers. Random Forest performed better.
    
    **Note**: the GridCV didn't work for multiple parameters and multiple values due to limitations 
    on training environment. 
    '''
    # create the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # add parameters. Extremely slow. doesn't work with multiple params and values
    
    parameters = {
        'clf__estimator__n_estimators': [10]
    }
    
    #GridCV to create model
    model = GridSearchCV(pipeline, parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - fitted ML model for evaluation
    X_test - numpy ndarray containing the predictor test dataset for model evaluation
    Y_test - numpy ndarray containing the target test dataset for model evaluation
    category_names - numpy ndarray containing the names of the target variable(s) 
    
    OUTPUT
    None - prints the evaluation report
    

    This function evaluates the model and prints the evaluation report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(np.hstack(Y_test),np.hstack(y_pred), target_names=category_names))


def save_model(model, model_filepath):
    '''
    INPUT
    model - final ML model to be saved
    model_filepath - string [path and filename of the pickle file to be saved]

    OUTPUT
    None
    

    This function saves the model as pickle file
    '''
    #save the model file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    file.close()


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
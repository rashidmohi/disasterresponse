import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - string [path and filename of messages csv data file]
    categories_filepath - string [path and filename of categories csv data file]

    OUTPUT
    df - pandas dataframe containing the combined data
    

    This function performs following steps to produce output dataframe:
    1. Load the messages and categories data
    2. Join the 2 datasets
    3. Create columns for the categories with their column names accordingly
    4. Replace the categories column with the new encoded categories columns
    '''
    #load the data files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df['categories']).str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    names_row = categories.iloc[0]


    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = names_row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames.tolist()
    
    #convert category columns to 1 and 0
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = pd.Series(categories[column]).apply(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # replace any value >1 to 1. Binary (0,1) values only
        categories.loc[categories[column] > 1, [column]] = 1
        
    # drop the original categories column from `df`
    # concatenate the original dataframe with the new `categories` dataframe
    df.drop(columns = 'categories',inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    return df

def clean_data(df):
    '''
    INPUT
    df- pandas dataframe containing the combined and original data
    

    OUTPUT
    df - andas dataframe containing the cleansed data (duplicate removed)
    

    This function removes the duplicate records and returns the cleansed dataset
    
    '''
    # drop duplicates
    df = df.drop_duplicates(keep = 'first')
    
    return df


def save_data(df, database_filename):
    '''
    INPUT
    df - pandas dataframe containing the final dataset to be stored in database
    database_filename - string [path and filename of target database file]

    OUTPUT
    None
    

    This function stores the final dataset into the given database
    '''
    #create the sql db and save the file
    db_file = 'sqlite:///'+database_filename
    engine = create_engine(db_file)
    df.to_sql('disaster_messages', engine, index=False, if_exists="replace")  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
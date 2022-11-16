import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Args:
    
    messages_filepath: str, the filepath to disaster_messages.csv
    categories_filepath: str, the filepath to disaster_categories.csv
    
    Returns a merged dataframe containing the data from the messages and categories files.
    
    Returns:
    
    df: pandas Dataframe, merged dataframe of messages and categories data.
    '''
    # Read in and merge csv datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on="id")
    
    print('Stage 1 passed: df merged successfully')
    
    return df


def clean_data(df):
    '''
    Args:
    
    df: pandas Dataframe, merged dataframe of messages and categories data.
    
    Performs necessary data cleaning and processing including:
     - splitting 'categories' column values to generate targets
     - renaming categories columns
     - replacing target values with 0, 1 for ML purposes
     - dropping unnecessary columns and duplicate rows
     
    Returns:
    
    df: pandas Dataframe, cleaned dataframe that can be used for ML.
     
    '''
    # Split categories column features into useable format
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    
    # Retrieve target category names
    get_names = lambda x: x[:-2]
    category_column_names = row.apply(get_names)
    categories.columns = category_column_names
    
    # Convert category row values to numeric format
    number_replace = lambda x: x.replace(x, x[-1])
    
    for column in categories:
        categories[column] = categories[column].apply(number_replace)
    
        categories[column] = categories[column].astype(int)
    
    # Drop unnecessary columns, duplicate rows, and concatenate for final df
    df = df.drop(columns='categories', axis=1)
    
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop_duplicates()
    
    # replacing target columns with erroneous value of 2 with value of 1 
    for name in category_column_names:
        df[name] = df[name].replace(2, 1)
    
    print('Stage 2 passed: df cleaned successfully')
    
    return df

def save_data(df, database_filename):
    '''
    Args:
    
    df: pandas Dataframe, cleaned dataframe that can be used for ML.
    database_filename: str, name of the SQL database
    
    Saves cleaned df into a SQLite database.
    
    Returns: None
    '''
    # Save cleaned data to SQLite database
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('DisasterResponsePipeline', engine, if_exists='replace',  index=False)
    
    print('Stage 3 passed: df saved successfully')

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

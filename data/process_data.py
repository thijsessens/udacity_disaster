import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''this function loads the data from csv path'''

    messages = pd.read_csv(messages_filepath)

    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories,on='id')
    
    return df


def clean_data(df):
    '''this function cleans the data'''

    categories = df.categories.str.split(";",expand=True)

    row = categories.iloc[0]

    category_colnames = row.str.slice(stop=-2)

    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.slice(start=-1)
        categories[column] = categories[column].astype('int32')

    df = df.drop(['categories'], axis=1)

    df = df.merge(categories,left_index=True,right_index=True)

    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''this function saves the data on the sqlite database'''

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('data', engine, index=False, if_exists='replace')  


def main():
    '''this is the main function'''

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

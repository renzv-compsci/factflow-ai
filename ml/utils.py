import pandas as pd 
import re 
from sklearn.model_selection import train_test_split


def load_data(fake_path='ml/data/training_set/Fake.csv', true_path='ml/data/training_set/True.csv'):
    """
    Load and combine fake and true news datasets
    Add label column: 0 for fake, 1 for true
    """
    fake_df = pd.read_csv(fake_path)
    fake_df ['label'] = 0
    true_df = pd.read_csv(true_path)
    true_df ['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    return df

def clean_text(text):
    """
    Basic text cleaning 
    -Lowercase
    -Remove non-alphabetic
    -Remove white space 
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text 

def preprocess_dataframe(df, text_column='text'):
    """
    Apply clean_text to a given column in the dataframe
    """
    df = df.dropna(subset=[text_column])
    df [text_column] = df[text_column].apply(clean_text)
    return df 

def split_data(df, test_size=0.2, random_state=42):
    """
    Split dataframe into train and test sets
    Return X_train, X_test, y_train, Y_train
    """
    X = df['text']
    y = df['label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_to_csv(df, filename):
    """
    Save dataframe to CSV
    """
    df.to_csv(filename, index=False)
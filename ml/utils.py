import pandas as pd 
import re 
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


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

def resample_balance(df: pd.DataFrame, label_col: str = 'binary_label', 
                     method: str = 'upsample', random_state: int = 42) -> pd.DataFrame:
    
    """
    Return a balanced copy of df by upsampling or downsampling classes.

    Args:
        df: input dataframe 
        label_col: name of the label column in df 
        method: 'upsample' | 'downsample'
        randome_state: seed for reproducibility
    """

    if label_col not in df.columns: 
        raise ValueError(f"label_col '{label_col}' not found in dataframe columns: {list(df.columns)}")
    
    counts = df[label_col].value_counts()
    if len(counts) <= 1:
        return df.copy()
    
    if method == 'upsample':
        target_n = counts.max()
        parts = []
        for cls, n in counts.items():
            cls_df = df[df[label_col] == cls]
            if n < target_n:
                cls_df = resample(cls_df, replace=True, n_samples=target_n, random_state=random_state)
            parts.append(cls_df)
        out = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
        return out 
    
    if method == 'downsample':
        tarhet_n = counts.min()
        parts = [] 
        for cls, n in counts.items():
            cls_df = df[df[label_col] == cls]
            if n > target_n:
                cls_df = resample(cls_df, replace=False, n_samples=target_n, random_state=random_state)
            parts.append(cls_df)
        out = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(droo=True)
        return out 
    
    raise ValueError("method must be 'upsample' or 'downsample'")

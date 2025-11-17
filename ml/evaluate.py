import argparse
import sys
import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.utils import resample

from utils import preprocess_dataframe

# try both relativbe and absoltue path import 
try: 
    from .utils import preprocess_dataframe,resample_balance
except Exception:
    from utils import preprocess_dataframe, resample_balance



def map_liar_label(label):
    """
    MAPS LIAR dataset style labels to binary classes:
    0 fake 1 true 
    """
    if pd.isnull(label):
        return None 
    label = str(label).lower().strip()
    if label in ['false', 'pants-fire', 'barely-true']:
        return 0
    elif label in ['true', 'mostly-true', 'half-true']:
        return 1
    else:
        return None 
    
def balance_classes(df, label_col):
    df_minority = df[df[label_col] == df[label_col].value_counts().idxmin()]
    df_majority = df[df[label_col] == df[label_col].value_counts().idxmax()]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    print(df['binary_label'].value_counts())

    return pd.concat([df_majority, df_minority_upsampled])

    
def check_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing: 
        raise ValueError(f"Missing columns: {missing}")

def check_numeric_columns(df, numeric_cols):
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric but got {df[col].dtype}")
    
def load_test_data(test_data_path, has_header=True, custom_text_col=None, custom_label_col=None):
    """
    Load a CSV/TSV test file and attempt to autodetect text and label columns.
    Returns: (df, text_col, label_col)
    Raises ValueError on missing columns.
    """
    # detect file format 
    sep = '\t' if test_data_path.endswith('.tsv') else ','
    header = 0 if has_header else None
    df = pd.read_csv(test_data_path, sep=sep, header=header)

    # if user passed numeric like strings for custom columns, convert to int 
    def _maybe_int(x):
        if x is None:
            return None 
        try: 
            return int(x)
        except Exception:
            return x
        
    custom_text_col = _maybe_int(custom_text_col)
    custom_label_col = _maybe_int(custom_label_col)

    text_col = custom_text_col
    label_col = custom_label_col

    if text_col is None: 
        possible_text_cols = ['text', 'content', 'article', 'statement', 'headline', 'body', 'news']
        text_col = next ((col for col in possible_text_cols if col in df.columns), None)
        if text_col is None and header is None:
            # if headerless and user didnt pass index default to column 0 as text 
            text_col = 0 if 0 in df.columns or 0 < df.shapep[1] else None
        if text_col is None:
            raise ValueError(f"No text column found. Columns available: {list(df.columns)}") 
        
    if label_col is None: 
        possible_label_cols = ['label', 'class', 'target', 'category', 'binary_label']
        label_col = next ((col for col in possible_label_cols if col in df.columns), None)
        if label_col is None and header is None: 
            # try last column as label
            label_col = df.columns[-1]
        if label_col is None: 
            raise ValueError(f"Non label column found. Coloumns available: {list(df.columns)}")
        
    return df, text_col, label_col


def main (test_data_paths, model_path='ml/models/logreg_pipeline.joblib', has_header=True, 
          custom_text_col=None, custom_label_col=None, balance_method='none', verbose=False):
    
    """
    Evaluate a saved pipeline on one or more 

    Args:
        test_data_paths = iterable of file path 
        model_path = path to joblib pipeline 
        has_header = wheter files have header row
        custom_text_col/custom_label_col: explicit column name or index
        balance_method: 'none'|'upsample'|'downsample' - only for analysis
        verbose: print debug info
    """
    

    # Load model
    try: 
        model = joblib.load(model_path) 
    except Exception as e: 
        raise RuntimeError(f"Failed to load model at '{model_path}': {e}")


    for test_data_path in test_data_paths: 
        print(f"\n=== Evaluating: {test_data_path}===")

        try: 
            df, text_col, label_col = load_test_data(test_data_path, has_header, custom_text_col, custom_label_col)
        except Exception as e: 
            print(f"Error loading '{test_data_path}': {e}")
            continue

        if verbose: 
            print("DEBUG - columns:", list(df.columns))
            print("DEBUG - sample rows:\n", df.head(3))

        # preprocess (preprocess_dataframe should handle concatenating text columns if needed)
        try: 
            df = preprocess_dataframe(df, text_column=text_col)
        except Exception as e:
            print(f"ERROR preprocessing '{test_data_path}': {e}")

        # balance (for analysis only)
        if balance_method and balance_method.lower() in ('upsample', 'downsample'):
            try: 
                df = resample_balance(df, label_col=label_col, method=balance_method.lower())
                if verbose:
                    print("DEBUG - balanced counts:\n", df[label_col].value_counts())
            except Exception as e: 
                print(f"WARNING - balancing failed : {e} (continuing without balancing)")  

        # Prepare inputs for prediction 
        if text_col not in df.columns and not isinstance(text_col, int):
            print(f"ERROR - expected text column '{text_col}' not found after preprocessing. Columns: {list(df.columns)}")
            continue

        X_test = df[text_col] if text_col in df.columns else df.iloc[:, int(text_col)]
        y_test = df[label_col] if label_col in df.columns else df.iloc[:, int(label_col)]
        
        # Predict
        try:
            y_pred = model.predict(X_test)
        except Exception as e: 
            print(f"ERROR during prediction on '{test_data_path}': {e}")
            continue 


        # map y_test 
        try:
            if str (label_col) == "binary_label" or int(label_col) == 13:
                y_test_mapped = pd.Series(y_test).astype(int)
                y_pred_mapped = pd.Series(y_pred).astype(int)
            else: 
                y_test_mapped = pd.Series(y_test).apply(map_liar_label)
                y_pred_mapped = pd.Series(y_pred).apply(map_liar_label)
        except Exception as e: 
            print(f"ERROR mapping labels: {e}")
            continue 

        # filter out rows
        valid_idx = (~y_test_mapped.isnull()) & (~pd.Series(y_pred_mapped).isnull())
        y_test_mapped = y_test_mapped[valid_idx]
        y_pred_mapped = pd.Series(y_pred_mapped)[valid_idx]

        if len(y_test_mapped) == 0:
            print("No valid samples after label mapping")
            continue
        
        labels = sorted(list(set(y_test_mapped)))
        print("Classification Report:")
        print(classification_report(y_test_mapped, y_pred_mapped, target_names=[str(l) for l in labels]))

        # confussion matrix 
        cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(l) for l in labels], yticklabels=[str(l) for l in labels])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {test_data_path}")
        plt.show()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test set(s).")
    parser.add_argument('test_data_paths', nargs='+', default='ml/data/test_set/train_mapped.tsv',help='Paths to the test data files (CSV or TSV)')
    parser.add_argument('--model_path', type=str, default='ml/models/logreg_pipeline.joblib', help='Path to trained model')
    parser.add_argument('--no_header', action='store_true', help='Indicate that the test data files DON\'T have headers')
    parser.add_argument('--text_col', type=str, default=None, help='Name of the text column if not auto-detectable')
    parser.add_argument('--label_col', type=str, default=None, help='Name of the label column if not auto-detectable')
    parser.add_argument('--balance-method', choices=['none', 'upsample', 'downsample'], default='none',
                        help='Optional resampling for analysis (does not change the trained model)')
    parser.add_argument('--verbose', action='store_true', help='Print debug info')
    args = parser.parse_args()

    main(
        args.test_data_paths,
        model_path=args.model_path,
        has_header=not args.no_header,
        custom_text_col=args.text_col,
        custom_label_col=args.label_col,
        balance_method=args.balance_method,
        verbose=args.verbose
    )
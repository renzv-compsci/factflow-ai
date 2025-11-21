import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

# try both relative and absolute imports for helpers
try:
    from .utils import preprocess_dataframe, resample_balance
except Exception:
    from .utils import preprocess_dataframe, resample_balance


def map_liar_label(label):
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
    print(df[label_col].value_counts())
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
    sep = '\t' if str(test_data_path).lower().endswith('.tsv') else ','
    header = 0 if has_header else None
    df = pd.read_csv(test_data_path, sep=sep, header=header)

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
        text_col = next((col for col in possible_text_cols if col in df.columns), None)
        if text_col is None and header is None:
            # headerless: default to first column index if available
            text_col = 0 if df.shape[1] >= 1 else None
        if text_col is None:
            raise ValueError(f"No text column found. Columns available: {list(df.columns)}")

    if label_col is None:
        possible_label_cols = ['label', 'class', 'target', 'category', 'binary_label']
        label_col = next((col for col in possible_label_cols if col in df.columns), None)
        if label_col is None and header is None:
            label_col = df.columns[-1]
        if label_col is None:
            raise ValueError(f"No label column found. Columns available: {list(df.columns)}")

    return df, text_col, label_col


def main(test_data_paths, model_path='ml/models/logreg_pipeline.joblib', has_header=True,
         custom_text_col=None, custom_label_col=None, balance_method='none', verbose=False):

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model at '{model_path}': {e}")

    for test_data_path in test_data_paths:
        print(f"\n=== Evaluating: {test_data_path} ===")

        try:
            df, text_col, label_col = load_test_data(test_data_path, has_header, custom_text_col, custom_label_col)
        except Exception as e:
            print(f"Error loading '{test_data_path}': {e}")
            continue

        if verbose:
            print("DEBUG - columns:", list(df.columns))
            print("DEBUG - sample rows:\n", df.head(3))

        try:
            # preprocess_dataframe expects a column label; if text_col is an index, convert it
            if isinstance(text_col, int):
                text_col_label = df.columns[int(text_col)]
            else:
                text_col_label = text_col
            df = preprocess_dataframe(df, text_column=text_col_label)
        except Exception as e:
            print(f"ERROR preprocessing '{test_data_path}': {e}")
            continue

        if balance_method and balance_method.lower() in ('upsample', 'downsample'):
            try:
                df = resample_balance(df, label_col=label_col, method=balance_method.lower())
                if verbose:
                    print("DEBUG - balanced counts:\n", df[label_col].value_counts())
            except Exception as e:
                print(f"WARNING - balancing failed : {e} (continuing without balancing)")

        # Prepare X_test and y_test using .iloc for index-based cols, column access otherwise
        try:
            if isinstance(text_col, int):
                X_test = df.iloc[:, int(text_col)]
            else:
                if text_col not in df.columns:
                    print(f"ERROR - expected text column '{text_col}' not found after preprocessing. Columns: {list(df.columns)}")
                    continue
                X_test = df[text_col]

            if isinstance(label_col, int):
                y_test_raw = df.iloc[:, int(label_col)]
            else:
                if label_col not in df.columns:
                    print(f"ERROR - expected label column '{label_col}' not found after preprocessing. Columns: {list(df.columns)}")
                    continue
                y_test_raw = df[label_col]
        except Exception as e:
            print(f"ERROR preparing inputs: {e}")
            continue

        # If X_test is a single-column DataFrame, convert to Series (pipeline expects 1D text input)
        if isinstance(X_test, pd.DataFrame) and X_test.shape[1] == 1:
            X_test = X_test.iloc[:, 0]

        # Predict
        try:
            y_pred_raw = model.predict(X_test)
        except Exception as e:
            print(f"ERROR during prediction on '{test_data_path}': {e}")
            continue

        # make series and map labels -> final series named y_test_s / y_pred_s
        try:
            y_test_s = pd.Series(y_test_raw).reset_index(drop=True)
            y_pred_s = pd.Series(y_pred_raw).reset_index(drop=True)

            if (isinstance(label_col, str) and str(label_col) == "binary_label") or (isinstance(label_col, int) and int(label_col) == 13):
                y_test_s = y_test_s.astype(int)
                y_pred_s = y_pred_s.astype(int)
            else:
                y_test_s = y_test_s.apply(map_liar_label)
                y_pred_s = y_pred_s.apply(map_liar_label)
        except Exception as e:
            print(f"ERROR mapping labels: {e}")
            continue

        # align and filter invalid mapped labels
        valid_idx = (~y_test_s.isnull()) & (~y_pred_s.isnull())
        y_test_s = y_test_s[valid_idx]
        y_pred_s = y_pred_s[valid_idx]

        if len(y_test_s) == 0:
            print("No valid samples after label mapping")
            continue

        labels = sorted(list(set(y_test_s)))
        print("Classification Report:")
        print(classification_report(y_test_s, y_pred_s, target_names=[str(l) for l in labels]))

        cm = confusion_matrix(y_test_s, y_pred_s, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(l) for l in labels], yticklabels=[str(l) for l in labels])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {test_data_path}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test set(s).")
    parser.add_argument('test_data_paths', nargs='+', default=['ml/data/test_set/train_mapped.tsv'], help='Paths to the test data files (CSV or TSV)')
    parser.add_argument('--model_path', type=str, default='ml/models/logreg_pipeline.joblib', help='Path to trained model')
    parser.add_argument('--no_header', action='store_true', help="Indicate that the test data files DON'T have headers")
    parser.add_argument('--text_col', type=str, default=None, help='Name or index of the text column if not auto-detectable')
    parser.add_argument('--label_col', type=str, default=None, help='Name or index of the label column if not auto-detectable')
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
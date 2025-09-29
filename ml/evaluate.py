import argparse
import sys
import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 

from utils import preprocess_dataframe

def map_liar_label(label):
    """
    MAPS LIAR dataset style labels to binary classes:
    0 fake 1 true 
    """
    label = str(label).lower().strip()
    if label in ['false', 'pants-fire', 'barely-true']:
        return 0
    elif label in ['true', 'mostly-true', 'half-true']:
        return 1
    else:
        return None 

def load_test_data(test_data_path, has_header=True, custom_text_col=None, custom_label_col=None):
    # detect file format 
    sep = '\t' if test_data_path.endswith('.tsv') else ','
    header = 0 if has_header else None

    df = pd.read_csv(test_data_path, sep=sep, header=header)

    # if no header, assign column names 
    if not has_header: 
        if custom_text_col is not None:
            try:
                custom_text_col = int(custom_text_col)
            except ValueError:
                pass

        if custom_label_col is not None: 
            try: 
                custom_label_col = int(custom_label_col)
            except ValueError:
                pass     
    # automatically find text and label column 
    text_col = custom_text_col
    label_col = custom_label_col

    if text_col is None:
        possible_text_cols = ['text', 'content', 'article', 'statement', 'headline', 'body', 'news']
        text_col = next((col for col in possible_text_cols if col in df.columns), None)
        if text_col is None:
            print(f"No text column found. Columns available: {df.columns}")
            sys.exit(1)

    if label_col is None:
        possible_label_cols = ['label', 'class', 'target', 'category']
        label_col = next((col for col in possible_label_cols if col in df.columns), None)
        if label_col is None:
            print(f"No label column found. Columns available: {df.columns}")
            sys.exit(1)

    return df, text_col, label_col

def main (test_data_paths, model_path='ml/models/logreg_model.joblib', vectorizer_path='ml/models/tfidf_vectorizer.joblib', has_header=True, custom_text_col=None, custom_label_col=None):

    # Load model & vectorizer   
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    for test_data_path in test_data_paths: 
        print(f"\n=== Evaluating: {test_data_path}===")

        df, text_col, label_col = load_test_data(test_data_path, has_header, custom_text_col, custom_label_col)
        df = preprocess_dataframe(df, text_column=text_col)
        X_test = df[text_col]
        y_test = df[label_col]

        # Map string labels to binary classes
        y_test_mapped = y_test.apply(map_liar_label)

        # Vectorize text
        X_test_tfidf = vectorizer.transform(X_test)

        # Predict
        y_pred = model.predict(X_test_tfidf)
        # Map predictions if model outputs string labels
        try:
            y_pred_mapped = pd.Series(y_pred).apply(map_liar_label)
        except Exception:
            y_pred_mapped = y_pred

        # Remove samples where label mapping failed (None)
        valid_idx = y_test_mapped.notnull()
        y_test_mapped = y_test_mapped[valid_idx]
        y_pred_mapped = pd.Series(y_pred_mapped)[valid_idx]

        # metrics 
        acc = accuracy_score(y_test_mapped, y_pred_mapped)
        print(f"Accuracy: {acc: .2f}\n")

        print("Classification report:")
        labels = sorted(list(set(y_test_mapped)))
        try:
            print(classification_report(y_test_mapped, y_pred_mapped, target_names=[str(l) for l in labels]))
        except Exception as e:
            print(classification_report(y_test_mapped, y_pred_mapped))
            print("Error in classification_report:", e)

        # for label names 
        labels = sorted(list(set(y_test_mapped)))
        try: 
            print(classification_report(y_test_mapped, y_pred_mapped, target_names=[str(1) for l in labels]))
        except Exception:
            print(classification_report(y_test_mapped, y_pred_mapped))

        # confussion matrix 
        cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(l) for l in labels], yticklabels=[str(l) for l in labels])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {test_data_path}")
        plt.show()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test set(s).")
    parser.add_argument('test_data_paths', nargs='+', help='Paths to the test data files (CSV or TSV)')
    parser.add_argument('--model_path', type=str, default='ml/models/logreg_model.joblib', help='Path to trained model file')
    parser.add_argument('--vectorizer_path', type=str, default='ml/models/tfidf_vectorizer.joblib', help='Path to vectorizer file')
    parser.add_argument('--no_header', action='store_true', help='Indicate that the test data files DON\'T have headers')
    parser.add_argument('--text_col', type=str, default=None, help='Name of the text column if not auto-detectable')
    parser.add_argument('--label_col', type=str, default=None, help='Name of the label column if not auto-detectable')
    args = parser.parse_args()
    main(
        args.test_data_paths,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        has_header=not args.no_header,
        custom_text_col=args.text_col,
        custom_label_col=args.label_col
    )
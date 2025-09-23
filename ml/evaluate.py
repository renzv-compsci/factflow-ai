import argparse
import sys
import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt 

from utils import preprocess_dataframe

def load_test_data(test_data_path='ml/data/test_set/train.tsv', has_header=True, custom_text_col=None, custom_label_col=None):
    # detect file format 
    sep = '\t' if test_data_path.endswith('.tsv') else ','
    header = 0 if has_header else None

    df = pd.read_csv(test_data_path, sep=sep, header=header)

    # if no header, assign column names 
    if not has_header: 
        df.columns = [
            'id', 'label', 'text', 'topic', 'speaker', 'job', 'state', 'party',
            'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context'
        ]

    # automatically find text and label column 
    text_col = custom_text_col
    label_col = custom_label_col

    if not text_col:
        possible_text_cols = ['text', 'content', 'article', 'statement', 'headline', 'body', 'news']
        text_col = next((col for col in possible_text_cols if col in df.columns), None)

        if text_col is None:
            print(f"No text column found. Columns available: {df.columns}")
            sys.exit(1)

    if not label_col:
        possible_label_cols = ['label', 'class', 'target', 'category']
        label_col = next((col for col in possible_label_cols if col in df.columns), None)

        if label_col is None:
            print(f"No label column found. Columns available: {df.columns}")
            sys.exit(1)

    return df, text_col, label_col

def main (test_data_paths, model_path='logreg_model.joblib', vectorizer_path='tfidf_vectorizer.joblib', has_header=True, custom_text_col=None, custom_label_col=None):

    # Load model & vectorizer   
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    for test_data_path in test_data_paths: 
        print(f"\n=== Evaluating: {test_data_path}===")

        df, text_col, label_col = load_test_data(test_data_path, has_header, custom_text_col, custom_label_col)
        df = preprocess_dataframe(df, text_column=text_col)
        X_test = df[text_col]
        y_test = df[label_col]

        # vectorize 
        X_test_tfidf = vectorizer.transform(X_test)

        # predict 
        y_pred = model.predict(X_test_tfidf)

        # metrics 
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc: .2f}\n")
        print("Classification report:")

        # for label names 
        labels = sorted(list(set(y_test)))
        try: 
            print(classification_report(y_test, y_pred, target_names=[str(1) for l in labels]))
        except Exception:
            print(classification_report(y_test, y_pred))

        # confussion matrix 
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(l) for l in labels], yticklabels=[str(l) for l in labels])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {test_data_path}")
        plt.show()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluate the trained model on test set(s).")
    parser.add_argument('test_data_paths', nargs='+', help='Paths to the test data files (CSV or TSV)')
    parser.add_argument('--model_path', type=str, default='logreg_model.joblib', help='Path to trained model file')
    parser.add_argument('--vectorizer_path', type=str, default='tfidf_vectorizer.joblib', help='Path to vectorizer file')
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
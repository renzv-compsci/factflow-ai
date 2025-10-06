# test minimal version of my pipeline and check if it produce 
# expected output types

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix


def make_sample_df():

    # Synthetic data 
    data = {
        '2': ["Fake news headline", "Real news headline", "Fake claim", "Real claim"],
        '3': ["politics", "health", "finance", "science"],
        '4': ["speaker1", "speaker2", "speaker3", "speaker4"],
        '5': ["title1", "title2", "title3", "title4"],
        '6': ["location1", "location2", "location3", "location4"],
        '7': ["party1", "party2", "party1", "party2"],
        '8': [1, 2, 3, 4],
        '9': [0.1, 0.2, 0.3, 0.4],
        '10': [5, 6, 7, 8],
        '11': [0, 1, 0, 1],
        '12': [2, 3, 4, 5],
        '13': ["context1", "context2", "context3", "context4"],
        'binary_label': [0, 1, 0, 1]
    }

    return pd.DataFrame(data)

def test_pipeline_runs():

    df = make_sample_df()
    text_cols = ['2', '3', '4', '5', '6', '7', '13']
    numeric_cols = ['8', '9', '10', '11', '12']
    df['combined_text'] = df.apply(lambda row: ' '.join([str(row[col]) for col in text_cols]), axis=1)
    X_numeric = df[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    y = df['binary_label']
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_text_tfidf = vectorizer.fit_transform(df['combined_text'])
    X_all = hstack([X_text_tfidf, csr_matrix(X_numeric_scaled)])
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    y_pred = cross_val_predict(model, X_all, y, cv=skf)
    report = classification_report(y, y_pred, output_dict=True)
    assert isinstance(report, dict)
    assert "0" in report and "1" in report
    assert len(y_pred) == len(y)

def test_vectorizer_not_empty():
    df = make_sample_df()
    text_cols = ['2', '3', '4', '5', '6', '7', '13']
    df['combined_text'] = df.apply(lambda row: ' '.join([str(row[col]) for col in text_cols]), axis=1)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_text_tfidf = vectorizer.fit_transform(df['combined_text'])
    assert X_text_tfidf.shape[0] == len(df)
    assert X_text_tfidf.shape[1] > 0  # Should have some features


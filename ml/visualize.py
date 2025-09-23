import joblib 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(vectorizer, model, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_positive_coefficients = coefs.argsort()[-top_n:]
    top_negative_coefficients = coefs.argsort()[:top_n]

    plt.figure(figsize=(10, 6))
    top_features = list(top_negative_coefficients) + list(top_positive_coefficients)
    colors = ['red' if c < 0 else 'blue' for c in coefs[top_features]]
    plt.barh(range(2 * top_n), coefs[top_features], colors=colors)
    plt.yticks(range(2 * top_n), [feature_names[i] for i in top_features])
    plt.title("Top Important Features (words)")
    plt.xlabel("Coefficient Value")
    plt.show()

if __name__ == "__main__":

    # Load model and vectorizer
    model = joblib.load('logreg_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    # Load and prepare test data
    fake_df = pd.read_csv('ml/data/training_set/Fake.csv')
    real_df = pd.read_csv('ml/data/training_set/True.csv')
    fake_df['label'] = 0
    real_df['label'] = 1
    df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=42)
    X = df['text'].fillna('')
    y = df['label']

    # Split data (same as in train.py)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_tfidf = vectorizer.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Show confusion matrix
    plot_confusion_matrix(y_test, y_pred, labels=['Fake', 'Real'])

    # Show classification report
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    # Show feature importance
    plot_feature_importance(vectorizer, model, top_n=15)
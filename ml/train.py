# Core Training WorkFlow 
import time
import joblib
from sklearn.calibration import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from utils import load_data, preprocess_dataframe, split_data

# Load and preprocess data
df = load_data('ml/data/training_set/Fake.csv', 'ml/data/training_set/True.csv')
df = preprocess_dataframe(df, text_column='text')
# Split data into train/test
X_train, X_test, y_train, y_test = split_data(df)

# use pipelines for a uniformed vectorizer for each model 
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Logistic regression with l1/l2 regularization search 
logreg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))
])

param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear'], 
}

grid = GridSearchCV(logreg_pipeline, param_grid, cv=skf, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate regularazation 
print("Best Logistic Regression params:", grid.best_params_)
print("Best F1 score:", grid.best_score_)
best_logreg = grid.best_estimator_

# save best logistic regression model and vectorizer together 
joblib.dump(best_logreg, 'logreg_pipeline.joblib')

# multi model comparison 
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', C=1, penalty='l2'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=10, random_state=42),
    "SVM (Linear)": LinearSVC(class_weight='balanced', max_iter=1000, verbose=1),
}

for name, model in models.items():
    print(f"Starting {name}...")
    start = time.time()
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('clf', model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')
    print(f"{name}: Mean F1 = {scores.mean():.3f} (+/-{scores.std():.3f}) (Time: {time.time() - start:.1f}s)")

# Evaluate the model
y_pred = best_logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}\n")
print("Classification report: ")
print(classification_report(y_test, y_pred, target_names = ['Fake', 'Real']))
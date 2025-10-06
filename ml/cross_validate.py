import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# Load Data
df = pd.read_csv('ml/data/test_set/train_mapped.tsv', sep='\t')
print(df.columns) # Debug 

# Combine relevant text columns (by index)
text_cols = ['2', '3', '4', '5', '6', '7', '13']
df['combined_text'] = df.apply(lambda row: ' '.join([str(row[col]) for col in text_cols]), axis=1)

# Numeric features
numeric_cols = ['8', '9', '10', '11', '12']
X_numeric = df[numeric_cols].fillna(0).values
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Target label
y = df['binary_label']

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_text_tfidf = vectorizer.fit_transform(df['combined_text'])

# Combine text and numeric features
X_all = hstack([X_text_tfidf, csr_matrix(X_numeric_scaled)])

# Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ML pipeline
pipeline = Pipeline([
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Hyperparameter grid
param_grid = {'clf__C': [0.01, 0.1, 1, 10]}

# Grid search for best parameters
grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring='f1', n_jobs=-1)
grid.fit(X_all, y)
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Cross-validated predictions, evaluation
y_pred = cross_val_predict(best_model, X_all, y, cv=skf)

print(classification_report(y, y_pred))
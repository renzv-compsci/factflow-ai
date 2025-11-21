# Create & Activate a python venv 
- python -m venv .venv 
- .\.venv\Script\Activate.ps1

# Install Requirements.txt 
- pip install -r requirements.txt

# If there is error, upgrade 
- python -m pip install --upgrade pip 

# Install essential package 
- pip install pandas scikit-learn joblib seaborn matplotlib pytest
- pip install opencv-python

# Running unit test 
- python -m pytest -q 

# Running evaluator and leakage checks 
- python -m ml.evaluate ml\data\test_set\train_mapped.tsv --model_path logreg_pipeline.joblib --label_col binary_label --text_col 2 --balance-method none --verbose
# End-to-End Machine Learning Project Guide

This guide walks through a structured workflow for a machine learning regression project (e.g., predicting California house prices). We cover **environment setup**, **project structure**, **data exploration**, **exception handling**, **logging**, **data ingestion**, **data transformation**, **model training**, **prediction pipeline**, and **Flask app integration**. Each section includes clear steps, example code, and explanations. All code blocks have inline comments and are properly formatted for clarity.

## Environment Setup

- **Create a new environment:** Use Conda or `venv` to isolate dependencies. For example:  
  ```bash
  conda create -n ml_env python=3.9 -y
  conda activate ml_env
  ```
- **Project files:** In the project root, create:
  - `README.md` – Overview of the project.
  - `requirements.txt` – List of Python dependencies.
  - `setup.py` – Installation and packaging script (if needed).

- **Example `requirements.txt`:** List all needed packages one per line. For example:  
  ```text
  pandas
  numpy
  seaborn
  matplotlib
  scikit-learn
  catboost
  xgboost
  dill
  flask
  -e .
  ```
  Here `-e .` allows installing the package in editable mode if you package this project.

- **Example `setup.py`:** This script reads `requirements.txt` and sets up the package. A sample structure is:  

  ```python
  from setuptools import find_packages, setup
  from typing import List

  def get_requirements(file_path: str) -> List[str]:
      """Read requirements from a file and return them as a list, excluding editable installs."""
      with open(file_path) as f:
          requirements = f.readlines()
      requirements = [req.strip() for req in requirements if req.strip()]
      # Remove '-e .' if present
      requirements = [req for req in requirements if req != '-e .']
      return requirements

  setup(
      name='california_house_price',
      version='0.0.1',
      author='Mithu',
      author_email='proshanta.mithu5@gmail.com',
      install_requires=get_requirements('requirements.txt'),
      packages=find_packages(),
  )
  ```
  - `get_requirements` reads and cleans each line, removing any `-e .` entry.
  - `setup()` uses `find_packages()` to include all modules under `src/`.

## Data Exploration

Before writing code, inspect your dataset in a Jupyter notebook or Python REPL:

- **Understand data schema:** Use `df.info()` to see data types and non-null counts.  
- **Summary statistics:** `df.describe()` to view basic statistics (mean, median, etc.).  
- **Missing values:** `df.isnull().sum()` helps identify columns with null entries.  
- **Feature correlation:** Use `df.corr()` and visualizations (like heatmaps) to see relationships.  
- **Drop unnecessary columns:** If a column is irrelevant or duplicates others, drop it: `df.drop(columns=['col1', 'col2'], inplace=True)`.  
- **Visualize data:** Plot distributions or scatterplots to understand target vs features.

This exploratory step helps identify which features are useful, potential preprocessing needs, and whether the problem is supervised (regression/classification) or unsupervised.

## Project Structure

```
project_root/
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/
│   │   ├── prediction_pipeline.py
│   │   ├── train_pipeline.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── predict_datapoint.html
│   │   ├── result.html
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── artifacts/
│   ├── (models, preprocessor, data files, logs)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Exception Handling (`src/exception.py`)

```python
import sys
from src.logger import logging

def error_message_details(error, error_details: sys) -> str:
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in script [{file_name}] "
        f"at line [{line_number}]: {error}"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details=sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details=error_details)

    def __str__(self):
        return self.error_message
```

---

## Logging Setup (`src/logger.py`)

```python
import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.log')
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

---

## Data Ingestion (`src/components/data_ingestion.py`)

```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

class DataIngestionConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path: str):
        logging.info("Starting data ingestion process.")
        try:
            df = pd.read_csv(file_path)
            logging.info("Dataset read into pandas DataFrame.")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}.")

            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data files created.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Exception occurred during data ingestion.")
            raise CustomException(e, sys)
```

---

## Data Transformation (`src/components/data_transformation.py`)

```python
import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = ['feature1', 'feature2']
            categorical_features = ['feature3']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            feature3_categories = ['category_low', 'category_medium', 'category_high']  
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(categories=[feature3_categories])),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numeric_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'target'
            input_train_df = train_df.drop(columns=[target_column], errors='ignore')
            target_train_df = train_df[target_column]
            input_test_df = test_df.drop(columns=[target_column], errors='ignore')
            target_test_df = test_df[target_column]

            preprocessing_obj = self.get_data_transformer_object()
            input_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_arr = preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            save_object(file_path=self.config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
```

---

## Utility Functions (`src/utils.py`)

```python
import os
import sys
import dill
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path: str, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    try:
        report = {}
        best_models = {}
        for model_name, model in models.items():
            param_grid = params.get(model_name, {})
            if param_grid:
                grid_search = GridSearchCV(
                    estimator=model, param_grid=param_grid,
                    cv=3, n_jobs=-1, scoring='r2'
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model
            best_models[model_name] = best_model
            y_test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_score
        return report, best_models
    except Exception as e:
        raise CustomException(e, sys)
```

Flask App Integration (app.py)

Create a simple Flask web app to serve the model. This app will have a homepage and a form to input data and display predictions.

from flask import Flask, request, render_template
import logging
from src.components.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    """
    Render the home page (index.html) with a button or link to start prediction.
    """
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    """
    GET: Render the input form (predict_datapoint.html).
    POST: Collect form data, make prediction, and display the result.
    """
    if request.method == 'GET':
        return render_template('predict_datapoint.html')
    else:
        # Collect input values from the form
        data = CustomData(
            MedInc=float(request.form.get('MedInc')),
            HouseAge=float(request.form.get('HouseAge')),
            AveRooms=float(request.form.get('AveRooms')),
            AveBedrms=float(request.form.get('AveBedrms')),
            Population=float(request.form.get('Population')),
            AveOccup=float(request.form.get('AveOccup'))
        )
        # Convert inputs to DataFrame and make prediction
        input_df = data.get_data_as_dataframe()
        logging.info(f"Input data for prediction: {input_df.to_dict(orient='records')}")
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_df)

        # Render the result page (result.html) with the prediction
        return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


Routes:

/ – Home page (index.html). Provide navigation or info about the app.

/predict_datapoint – Handles both showing the form (GET) and processing submissions (POST). The HTML form (predict_datapoint.html) should have input fields named MedInc, HouseAge, etc., matching request.form.get(...) keys.

Templates: You need corresponding HTML files in src/templates/:

index.html – A simple welcome page with a link/button to the prediction form.

predict_datapoint.html – A form where users enter feature values. The form’s action should point to /predict_datapoint and method="post".

result.html – Displays the prediction result (passed as prediction context variable).

Logging: We log input data for debugging.

With the Flask app running, you can navigate to http://localhost:5000/ to access the UI, enter values, and view predictions.

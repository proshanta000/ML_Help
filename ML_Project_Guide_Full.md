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

    # This function reads the requirements from the 'requirements.txt' file.
    def get_requirements(file_path: str) -> List[str]:
        """
        Reads requirements from a given file path and returns a list of strings,
        excluding editable installs like '-e .'.
        """
        with open(file_path) as f:
            # Read all lines from the file
            requirements = f.readlines()

        # Remove any whitespace characters like '\n' at the end of each line
        requirements = [req.strip() for req in requirements if req.strip()]

        # Filter out the editable install flag '-e .'
        requirements = [req for req in requirements if req != '-e .']
        
        return requirements

    # The setup() function is the main part of the script.
    # It uses the setuptools library to build and package the project.
    setup(
        # The name of your package
        name='california_house_price',
        
        # The version of your package
        version='0.0.1',
        
        # The name of the author
        author='Mithu',
        
        # The author's email
        author_email='proshanta.mithu5@gmail.com',
        
        # The list of packages required for the project to run.
        # We call the get_requirements() function to get this list from requirements.txt.
        install_requires=get_requirements('requirements.txt'),
        
        # Automatically finds and includes all packages in the 'src' directory.
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
    """
    Constructs a detailed error message including the script file name and line number
    where the exception occurred.
    """
    # Get traceback information from the sys module
    _, _, exc_tb = error_details.exc_info()
    
    # Extract the file name from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Extract the line number from the traceback object
    line_number = exc_tb.tb_lineno
    
    # Format the error message to be more informative
    error_message = (
        f"Error occurred in script [{file_name}] "
        f"at line [{line_number}]: {error}"
    )
    
    return error_message

class CustomException(Exception):
    """
    A custom exception class that inherits from the base Exception class.
    It provides a more detailed and user-friendly error message.
    """
    def __init__(self, error_message, error_details=sys):
        # Call the constructor of the parent class (Exception)
        super().__init__(error_message)
        
        # Store the detailed error message using our helper function
        self.error_message = error_message_details(error_message, error_details=error_details)

    def __str__(self):
        """
        Returns the custom error message when the exception is printed.
        """
        return self.error_message

```

---

## Logging Setup (`src/logger.py`)

```python
import logging
import os
from datetime import datetime

# Define the log directory path relative to the current working directory.
LOG_DIR = os.path.join(os.getcwd(), "logs")

# Create the log directory if it does not already exist.
os.makedirs(LOG_DIR, exist_ok=True)

# Create a unique log file name using the current timestamp.
# The format is YYYY_MM_DD_HH_MM_SS.log
LOG_FILE = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.log')

# Construct the full path for the log file.
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure the basic settings for logging.
logging.basicConfig(
    # Set the filename where log messages will be written.
    filename=LOG_FILE_PATH,
    
    # Define the format of each log message.
    # [%(asctime)s] - Timestamp
    # %(lineno)d - Line number
    # %(name)s - Logger name
    # %(levelname)s - Log level (e.g., INFO, ERROR)
    # %(message)s - The log message content
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    
    # Set the logging level. Only messages with this level or higher will be logged.
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
    """
    A class to hold configuration paths for data ingestion.
    This uses a dataclass-like structure to make the paths easily accessible.
    """
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    """
    Handles the data ingestion process, including reading the dataset
    and splitting it into training and testing sets.
    """
    def __init__(self):
        # Initialize the configuration with the defined paths.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path: str):
        """
        Reads the dataset from the provided file path, saves a raw copy,
        and splits it into training and testing CSV files.
        
        Args:
            file_path (str): The path to the raw dataset file.

        Returns:
            tuple: A tuple containing the paths to the train and test data files.
        """
        logging.info("Starting data ingestion process.")
        try:
            # Read the raw data from the specified file path into a DataFrame.
            df = pd.read_csv(file_path)
            logging.info("Dataset read into pandas DataFrame.")

            # Create the 'artifacts' directory if it doesn't exist to store data files.
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw data to a CSV file in the 'artifacts' directory for reference.
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}.")

            # Split the data into training (75%) and testing (25%) sets.
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
            
            # Save the training set to a CSV file.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Save the testing set to a CSV file.
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data files created.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Exception occurred during data ingestion.")
            # Raise a custom exception for better error tracking.
            raise CustomException(e, sys)

# This block allows the script to be run directly for testing.
if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        # Call the ingestion process with the path to your raw data file.
        data_ingestion.initiate_data_ingestion("data/california_housing.csv")
    except Exception as e:
        logging.error("Data ingestion failed.")
        # Raise a custom exception if the process fails.
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
    """
    Holds the configuration paths for saving the preprocessor object.
    """
    def __init__(self):
        # The file path where the preprocessor object will be saved.
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Handles the entire data transformation process, including creating preprocessing pipelines
    and applying them to the data.
    """
    def __init__(self):
        # Initialize the configuration object.
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a data transformation object. This object is a ColumnTransformer
        that applies different preprocessing steps to numeric and categorical features.
        """
        try:
            # Placeholder feature lists. You must replace these with the actual column names from your dataset.
            numeric_features = ['feature1', 'feature2']
            categorical_features = ['feature3']

            # Define the pipeline for numeric features.
            num_pipeline = Pipeline(steps=[
                # Step 1: Impute missing values with the median.
                ('imputer', SimpleImputer(strategy='median')),
                # Step 2: Scale the features to have a mean of 0 and a variance of 1.
                ('scaler', StandardScaler())
            ])

            # Define the categories for the ordinal encoder.
            feature3_categories = ['category_low', 'category_medium', 'category_high']
            
            # Define the pipeline for categorical features.
            cat_pipeline = Pipeline(steps=[
                # Step 1: Impute missing values with the most frequent value.
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # Step 2: Encode the categorical features as ordered integers.
                ('ordinal_encoder', OrdinalEncoder(categories=[feature3_categories])),
                # Step 3: Scale the encoded features.
                ('scaler', StandardScaler())
            ])

            # Combine the pipelines into a single preprocessor using ColumnTransformer.
            # This applies the correct pipeline to the correct set of columns.
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numeric_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])
            return preprocessor
        except Exception as e:
            # Raise a custom exception if an error occurs during the process.
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Loads the training and testing data, applies the preprocessing steps,
        and returns the transformed data arrays and the preprocessor file path.
        
        Args:
            train_path (str): The file path to the training dataset.
            test_path (str): The file path to the testing dataset.

        Returns:
            tuple: A tuple containing the transformed training array, testing array,
                   and the file path of the saved preprocessor object.
        """
        try:
            # Read the training and testing data from the provided paths.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Define the target column name.
            target_column = 'target'

            # Separate features (input) and target (output) for both train and test data.
            input_train_df = train_df.drop(columns=[target_column], errors='ignore')
            target_train_df = train_df[target_column]
            input_test_df = test_df.drop(columns=[target_column], errors='ignore')
            target_test_df = test_df[target_column]

            # Get the preprocessor object.
            preprocessing_obj = self.get_data_transformer_object()
            
            # Fit the preprocessor on the training data and transform it.
            input_train_arr = preprocessing_obj.fit_transform(input_train_df)
            
            # Transform the testing data using the fitted preprocessor.
            input_test_arr = preprocessing_obj.transform(input_test_df)

            # Combine the transformed feature arrays with the target arrays.
            # np.c_ is used to concatenate arrays along the second axis (columns).
            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            # Save the fitted preprocessor object to a file for later use in prediction.
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            # Raise a custom exception if an error occurs during transformation.
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
    """
    Saves a Python object to a file using the dill library.
    
    This function is robust, automatically creating the necessary
    directories if they don't already exist before saving the object.

    Args:
        file_path (str): The full path, including the filename, where the object will be saved.
        obj: The Python object to be saved.
    """
    try:
        # Get the directory name from the provided file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it does not exist. The 'exist_ok=True'
        # argument prevents an error if the directory already exists.
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode ('wb') and dump the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object successfully saved to {file_path}")

    except Exception as e:
        # If an error occurs, wrap it in a CustomException for consistent error handling
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    Loads a Python object from a file using the dill library.

    Args:
        file_path (str): The full path to the file from which the object will be loaded.

    Returns:
        obj: The Python object loaded from the file.
    """
    try:
        # Open the file in binary read mode ('rb') and load the object
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        
        logging.info(f"Object successfully loaded from {file_path}")
        return obj

    except Exception as e:
        # If an error occurs, wrap it in a CustomException
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    """
    Trains and evaluates a dictionary of machine learning models.
    
    This function iterates through each model, performs hyperparameter tuning
    if a parameter grid is provided, and then evaluates the best model's
    performance on the test set.

    Args:
        X_train: The training feature data.
        y_train: The training target data.
        X_test: The testing feature data.
        y_test: The testing target data.
        models (dict): A dictionary where keys are model names (str) and values are model instances.
        params (dict): A dictionary where keys are model names (str) and values are
                       the hyperparameter grids (dict) for GridSearchCV.

    Returns:
        tuple: A tuple containing two dictionaries:
               - report (dict): Model names mapped to their R² scores on the test set.
               - best_models (dict): Model names mapped to their best fitted model instances.
    """
    try:
        report = {}
        best_models = {}
        
        # Iterate over each model provided in the dictionary
        for model_name, model in models.items():
            logging.info(f"Starting evaluation for model: {model_name}")
            
            # Check if a parameter grid exists for the current model
            param_grid = params.get(model_name, {})
            
            if param_grid:
                # If a parameter grid is provided, use GridSearchCV for tuning
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,            # Use 3-fold cross-validation
                    n_jobs=-1,       # Use all available CPU cores
                    scoring='r2'     # Use R² score as the evaluation metric
                )
                
                # Fit the grid search object to the training data to find the best model
                grid_search.fit(X_train, y_train)
                
                # Get the best model instance from the grid search
                best_model = grid_search.best_estimator_
                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

            else:
                # If no parameter grid is provided, train the model directly
                model.fit(X_train, y_train)
                best_model = model
            
            # Store the best fitted model instance
            best_models[model_name] = best_model
            
            # Make predictions on the test data using the best model
            y_test_pred = best_model.predict(X_test)
            
            # Calculate the R² score to evaluate performance
            test_score = r2_score(y_test, y_test_pred)
            
            # Store the R² score in the report dictionary
            report[model_name] = test_score
            logging.info(f"Evaluation for {model_name} completed. R² Score: {test_score:.4f}")

        return report, best_models
        
    except Exception as e:
        # If an error occurs during evaluation, raise a custom exception
        raise CustomException(e, sys)

```

---

## Model Training (`src/components/model_trainer.py`)

```python
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Get the absolute path of the current script's directory.
# This ensures that no matter where you run the script from,
# it can always find the project's root directory.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root directory.
# For example: E:/Data_science/Projects/california_house_price
project_root = os.path.join(script_dir, '..')

# Add the project root to the system path
# This allows Python to find and import modules from the 'src' directory.
sys.path.append(project_root)

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_models


@dataclass
class ModelTraningConfig:
    """
    Configuration for model training path and parameters
    """
    # Define the path where the trained model will be saved.
    # It's set to 'artifacts/model.pkl'.
    traning_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    Class for training machine learning models
    """
    def __init__(self):
        # Initialize the configuration object for the model trainer.
        self.model_traning_config = ModelTraningConfig()

    def initiate_model_traning(self, train_array, test_array):
        """
        This method trains multiple regression models and evaluates their performance.
        It returns the best model based on r2 score.
        """
        try:
            logging.info("Entering the model training method or component.")

            # Fix for the CatBoostError: create an absolute path for the writable directory
            # CatBoost requires a directory to save intermediate files. This line creates one
            # inside the 'artifacts' directory.
            catboost_dir = os.path.join(os.getcwd(), 'artifacts', 'catboost_info')
            os.makedirs(catboost_dir, exist_ok=True)
            logging.info(f"Created directory for CatBoost: {catboost_dir}")

            logging.info("Splitting the training and testing input and target variables.")
            # Splitting the train and test arrays into features and target variables.
            # The last column is assumed to be the target variable.
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Here all is for supervised data
            # Creating models dictionary with various regression models.
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
                'XGB Regressor': XGBRegressor(random_state=42),
                'KNeighbors Regressor': KNeighborsRegressor(),
                # Fix for the CatBoostError: specify a writable directory
                # The train_dir parameter is set to the created directory to prevent errors.
                'CatBoost Regressor': CatBoostRegressor(verbose=False, random_state=42, train_dir=catboost_dir),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42)
            }

            # Define parameters grid for each model.
            # IMPORTANT: Adjust these parameters and ranges based on dataset and computational resources.
            # Start with broader ranges and narrow down as results come.
            params = {
                'Linear Regression': {},  # No hyperparameters for basic Linear Regression in this context
                'Decision Tree Regressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [10, 20, None],  # Adjusted range for faster execution
                    'min_samples_split': [2, 5],  # Adjusted range
                    'min_samples_leaf': [1, 2]  # Adjusted range
                },
                'Random Forest Regressor': {
                    'n_estimators': [64, 128, 256],  # Adjusted range
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [10, 20, None],  # Adjusted range
                    'min_samples_split': [2, 5],  # Adjusted range
                    'min_samples_leaf': [1, 2]  # Adjusted range
                },
                'Gradient Boosting Regressor': {
                    'learning_rate': [.1, .01, .05],  # Adjusted range
                    'subsample': [0.7, 0.8, 0.9],  # Adjusted range
                    'n_estimators': [64, 128, 256],  # Adjusted range
                    'max_depth': [3, 5]  # Adjusted range
                },
                'XGB Regressor': {
                    'learning_rate': [.1, .01, .05],  # Adjusted range
                    'n_estimators': [64, 128, 256],  # Adjusted range
                    'max_depth': [3, 5],  # Adjusted range
                    'subsample': [0.7, 0.8],  # Adjusted range
                    'colsample_bytree': [0.7, 0.8],  # Adjusted range
                    'gamma': [0, 0.1]  # Adjusted range
                },
                'KNeighbors Regressor': {
                    'n_neighbors': [5, 7, 9],  # Adjusted range
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree']  # Adjusted range
                },
                'CatBoost Regressor': {
                    'depth': [6, 8],  # Adjusted range
                    'learning_rate': [0.01, 0.05],  # Adjusted range
                    'iterations': [50, 100]  # Adjusted range
                },
                'AdaBoost Regressor': {
                    'learning_rate': [.1, .01, 0.5],  # Adjusted range
                    'n_estimators': [64, 128, 256]  # Adjusted range
                }
            }

            # Call the updated evaluate models function from `src.utils`.
            # This function is expected to train and evaluate each model with its parameters.
            model_reports, tuned_models = evalute_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Finding the best model based on r2 score from the models report dictionary.
            best_model_score = max(sorted(model_reports.values()))

            # Finding the best model name from model report dictionary.
            # This line gets the key (model name) corresponding to the highest R2 score.
            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ]

            # Retrieve the actual best tuned model instance from the `tuned_models` dictionary.
            best_model = tuned_models[best_model_name]

            # If the best model score is less than 0.6, raise a custom exception.
            # This acts as a quality gate for the model.
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy.", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score:.4f}")

            # Save the best model object to a pickle file using a utility function.
            save_object(
                file_path=self.model_traning_config.traning_model_file_path,
                obj=best_model
            )

            # This is the new line that returns the best model and its score.
            return best_model_name, best_model_score, best_model

        except Exception as e:
            logging.info("Exception occurred in the initiate_model_training.")
            # Raise a custom exception for better error handling and traceability.
            raise CustomException(e, sys)
```
```python
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Get the absolute path of the current script's directory.
# This ensures that no matter where you run the script from,
# it can always find the project's root directory.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root directory.
# For example: E:/Data_science/Projects/california_house_price
project_root = os.path.join(script_dir, '..')

# Add the project root to the system path
# This allows Python to find and import modules from the 'src' directory.
sys.path.append(project_root)

# Importing the necessary classification models from scikit-learn, CatBoost, and XGBoost.
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Importing custom utility functions and classes for error handling and logging.
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_models


@dataclass
class ModelTraningConfig:
    """
    Configuration for model training path and parameters.
    This class is used to store configuration variables, making them easy to access.
    """
    # Define the path where the trained model will be saved as a pickle file.
    traning_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    Class for training machine learning models.
    This class encapsulates the entire model training process, including evaluation and saving.
    """
    def __init__(self):
        # Initialize the configuration object when an instance of ModelTrainer is created.
        self.model_traning_config = ModelTraningConfig()

    def initiate_model_traning(self, train_array, test_array):
        """
        This method trains multiple classification models and evaluates their performance.
        It returns the best model based on the accuracy score.

        Args:
            train_array (np.ndarray): The training data array (features and target).
            test_array (np.ndarray): The testing data array (features and target).

        Returns:
            tuple: A tuple containing the best model's name, its accuracy score, and the model object.
        """
        try:
            logging.info("Entering the model training method or component.")

            # Fix for the CatBoostError: create an absolute path for the writable directory.
            # CatBoost requires a directory to save intermediate files. This line ensures it's created
            # and writable within the project structure.
            catboost_dir = os.path.join(os.getcwd(), 'artifacts', 'catboost_info')
            os.makedirs(catboost_dir, exist_ok=True)
            logging.info(f"Created directory for CatBoost: {catboost_dir}")

            logging.info("Splitting the training and testing input and target variables.")
            # Splitting the train and test arrays into features (X) and target (y).
            # The last column is assumed to be the target variable.
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Creating a dictionary of different classification models to be trained.
            # These are the base model instances before hyperparameter tuning.
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
                'Random Forest Classifier': RandomForestClassifier(random_state=42),
                'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
                'XGB Classifier': XGBClassifier(random_state=42),
                'KNeighbors Classifier': KNeighborsClassifier(),
                # The train_dir parameter is specified to the created directory to prevent errors.
                'CatBoost Classifier': CatBoostClassifier(verbose=False, random_state=42, train_dir=catboost_dir),
                'AdaBoost Classifier': AdaBoostClassifier(random_state=42)
            }

            # Define the parameter grids for hyperparameter tuning for each model.
            # IMPORTANT: Adjust these parameters and ranges based on dataset and computational resources.
            params = {
                'Logistic Regression': {
                    'solver': ['liblinear', 'lbfgs'],
                    'C': [0.1, 1.0, 10.0]
                },
                'Decision Tree Classifier': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'Random Forest Classifier': {
                    'n_estimators': [100, 200, 300],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'Gradient Boosting Classifier': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5],
                    'subsample': [0.7, 0.9, 1.0]
                },
                'XGB Classifier': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5],
                    'subsample': [0.7, 0.9, 1.0],
                    'gamma': [0, 0.1]
                },
                'KNeighbors Classifier': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                'CatBoost Classifier': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200]
                },
                'AdaBoost Classifier': {
                    'learning_rate': [0.01, 0.1, 1.0],
                    'n_estimators': [50, 100, 200]
                }
            }

            # Call the `evalute_models` function to train and evaluate all models.
            # This function is expected to return a report of accuracy scores and the tuned model objects.
            model_reports, tuned_models = evalute_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Find the best accuracy score from the evaluation reports.
            best_model_score = max(sorted(model_reports.values()))

            # Find the name of the model that achieved the best accuracy score.
            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ]

            # Retrieve the actual best tuned model instance from the dictionary.
            best_model = tuned_models[best_model_name]

            # If the best model's accuracy score is less than a certain threshold (0.6), raise an exception.
            # This acts as a quality check for the model's performance.
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy.", sys)

            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score:.4f}")

            # Save the best model to a file using the `save_object` utility function.
            save_object(
                file_path=self.model_traning_config.traning_model_file_path,
                obj=best_model
            )

            # Return the results of the training process.
            return best_model_name, best_model_score, best_model
            
        except Exception as e:
            logging.info("Exception occurred in the initiate_model_training.")
            # Raise a custom exception for better error handling and traceability.
            raise CustomException(e, sys)
```
```python
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Get the absolute path of the current script's directory.
# This ensures that no matter where you run the script from,
# it can always find the project's root directory.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root directory.
# For example: E:/Data_science/Projects/california_house_price
project_root = os.path.join(script_dir, '..')

# Add the project root to the system path
# This allows Python to find and import modules from the 'src' directory.
sys.path.append(project_root)

# Importing the necessary unsupervised models from scikit-learn.
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

# Importing custom utility functions and classes for error handling and logging.
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_models


@dataclass
class ModelTraningConfig:
    """
    Configuration for model training path and parameters.
    This class is used to store configuration variables, making them easy to access.
    """
    # Define the path where the trained model will be saved as a pickle file.
    traning_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    Class for training machine learning models.
    This class encapsulates the entire model training process, including evaluation and saving.
    """
    def __init__(self):
        # Initialize the configuration object when an instance of ModelTrainer is created.
        self.model_traning_config = ModelTraningConfig()

    def initiate_model_traning(self, train_array, test_array):
        """
        This method trains multiple unsupervised models.
        
        NOTE: Unsupervised models do not use a target variable (`y`) for training,
        and their evaluation metrics differ from supervised models. The current
        `evalute_models` function and the subsequent best model selection logic
        are designed for supervised learning and will not work correctly for this task.
        These parts of the code would need to be re-implemented for unsupervised
        model evaluation (e.g., using a Silhouette score for clustering).

        Args:
            train_array (np.ndarray): The training data array (features only).
            test_array (np.ndarray): The testing data array (features only).

        Returns:
            tuple: A tuple containing the best model's name, its accuracy score, and the model object.
        """
        try:
            logging.info("Entering the model training method or component.")

            # Fix for the CatBoostError: create an absolute path for the writable directory.
            # CatBoost requires a directory to save intermediate files. This line ensures it's created
            # and writable within the project structure.
            catboost_dir = os.path.join(os.getcwd(), 'artifacts', 'catboost_info')
            os.makedirs(catboost_dir, exist_ok=True)
            logging.info(f"Created directory for CatBoost: {catboost_dir}")

            logging.info("Splitting the training and testing input and target variables.")
            # Splitting the train and test arrays into features and target variables
            # NOTE: Unsupervised models do not use a target variable.
            # For this reason, the `y_train` and `y_test` variables are not needed for fitting the models.
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Here for unsupervised data
            # Creating a dictionary with various unsupervised models to be trained.
            models = {
                'KMeans': KMeans(random_state=42, n_init='auto'),
                'DBSCAN': DBSCAN(),
                'Agglomerative Clustering': AgglomerativeClustering(),
                'PCA': PCA(random_state=42),
                'TSNE': TSNE(random_state=42, init='pca'),
                'Isolation Forest': IsolationForest(random_state=42)
            }

            # Define parameters grid for each model
            # IMPORTANT: Adjust these parameters and ranges based on dataset and computational resources.
            params = {
                'KMeans': {
                    'n_clusters': [3, 5, 8],
                    'init': ['k-means++', 'random']
                },
                'DBSCAN': {
                    'eps': [0.3, 0.5, 0.7],
                    'min_samples': [5, 10, 20]
                },
                'Agglomerative Clustering': {
                    'n_clusters': [3, 5, 8],
                    'linkage': ['ward', 'complete', 'average', 'single']
                },
                'PCA': {
                    'n_components': [2, 5, 10],
                    'svd_solver': ['full', 'arpack', 'randomized']
                },
                'TSNE': {
                    'n_components': [2, 3],
                    'perplexity': [5, 30, 50],
                    'learning_rate': [200, 500, 1000]
                },
                'Isolation Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_features': [1.0, 0.8, 0.6],
                    'contamination': ['auto', 0.05, 0.1]
                }
            }
            
            # NOTE: The following section requires the `evalute_models` function to be
            # adapted for unsupervised learning. It currently expects a target variable (`y`)
            # and a performance metric like r2_score. This will need to be changed to
            # use an appropriate unsupervised metric, such as Silhouette score for clustering.
            
            # Call the updated evalute models function
            model_reports, tuned_models = evalute_models(
                X_train= X_train, 
                y_train= y_train,
                X_test= X_test,
                y_test= y_test,
                models= models,
                params= params
            )
            
            # Finding the best model based on r2 score from the models report dictionary
            best_model_score = max(sorted(model_reports.values()))

            # Finding the best model name from model report dictionary 
            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ]

            # Retrieve the actual best tuned model instance
            best_model = tuned_models[best_model_name]

            # If the best model score is less than 0.6, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy.", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_traning_config.traning_model_file_path,
                obj=best_model
            )

            # This is the new line that returns the best model and its score
            return best_model_name, best_model_score, best_model
            
        except Exception as e:
            logging.info("Exception occurred in the initiate_model_training.")
            raise CustomException(e, sys)

```

---

## Prediction Pipeline (`src/pipeline/prediction_pipeline.py`)

```python
import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipline:
    """
    Class for making predictions using a trained model and preprocessor.
    """
    def __init__(self):
        """
        Initializes the prediction pipeline by loading the model and preprocessor.
        The paths are hardcoded as they are assumed to be in the 'artifacts' directory.
        """
        # Define the file paths for the saved model and preprocessor.
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        # Load the saved model and preprocessor objects from their respective paths.
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)

    def predict(self, feature):
        """
        Makes a prediction using the loaded model and preprocessor.

        Args:
            feature (pd.DataFrame): The input features for which to make a prediction.

        Returns:
            np.ndarray: The predicted value(s).
        """
        try:
            # Preprocess the input features using the loaded preprocessor.
            data_scaled = self.preprocessor.transform(feature)
            logging.info('Data Preprocessing completed')

            # Make a prediction using the loaded model.
            prediction = self.model.predict(data_scaled)
            logging.info("Prediction completed")

            return prediction

        except Exception as e:
            # Raise a custom exception if any error occurs during prediction.
            raise CustomException(e, sys)


class CustomData:
    """
    Class for handling custom data input for prediction.
    It structures the raw data into a format suitable for the prediction pipeline.
    """
    def __init__(self, MedInc: float, HouseAge: float):
        """
        Initializes the CustomData object with input features.
        
        Args:
            MedInc (float): The median income. (This is a demo feature.)
            HouseAge (float): The house age. (This is a demo feature.)
        """
        # Assign the input features to instance variables.
        self.MedInc = MedInc
        self.HouseAge = HouseAge
        # Note: You should update this class to match the features in your actual dataset.
    
    def get_data_as_dataframe(self):
        """
        Converts the custom data instance into a pandas DataFrame.
        This is a crucial step as most machine learning models are trained
        to accept input in a DataFrame format.
        
        Returns:
            pd.DataFrame: A DataFrame containing the custom data.
        """
        try:
            # Create a dictionary from the instance variables.
            custom_data_input_dict = {
                "MedInc": [self.MedInc],
                "HouseAge": [self.HouseAge],
            }
            # Convert the dictionary to a pandas DataFrame and return it.
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            # Raise a custom exception if there is an error in data conversion.
            logging.error("Error occurred while creating DataFrame from custom data.")
            raise CustomException(e, sys)

```

---

## Flask App Integration (`app.py`)

```python
import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# This section is for setting up the Python path to allow
# for importing modules from different directories in the project.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(project_root)

# Import the custom classes needed for the prediction pipeline.
from src.pipeline.predict_pipeline import PredictPipline, CustomData


# Initialize the Flask application
application = Flask(__name__) 

# Assign the Flask application instance to a variable named 'app'
# This is a common practice for some hosting services.
app = application 

# Route for the home page
@app.route('/') 
def home():
    """
    Renders the main input form page (index.html).
    This is the first page the user sees when they navigate to the root URL.
    """
    return render_template('index.html')

# This link is part of the HTML in index.html, not Python.
# It directs the user to the prediction page.
# <a href="{{ url_for('predict_datapoint') }}" class="inline-block w-full py-4 px-6 bg-blue-600 text-white font-bold text-lg rounded-full shadow-lg hover:bg-blue-700 transition duration-300 transform hover:scale-105">
#     Start Your Prediction
# </a>

# Route to handle both displaying the form and processing the prediction
@app.route('/predict_datapoint', methods=['GET', 'POST']) 
def predict_datapoint():
    """
    Handles displaying the prediction form (GET) and processing
    the form submission (POST) to make a prediction.
    """
    if request.method == 'GET':
        # If the request is a GET, render the form page where the user can input data.
        return render_template('predict_datapoint.html')
    else:
        # If the request is a POST (form submission), process the data.
        # Create a CustomData object from form inputs.
        # The form fields in predict_datapoint.html must match these names exactly.
        data = CustomData(
            MedInc=float(request.form.get('MedInc')),
            HouseAge=float(request.form.get('HouseAge')),
            AveRooms=float(request.form.get('AveRooms')),
            AveBedrms=float(request.form.get('AveBedrms')),
            Population=float(request.form.get('Population')),
            AveOccup=float(request.form.get('AveOccup')),
            Latitude=float(request.form.get('Latitude')),
            Longitude=float(request.form.get('Longitude'))
        )
        
        # Convert CustomData to a DataFrame for the prediction pipeline
        pred_df = data.get_data_as_dataframe()
        print(pred_df) # For debugging, prints the DataFrame to the console.

        # Initialize and run your prediction pipeline
        predict_pipeline = PredictPipline()
        results = predict_pipeline.predict(pred_df)
        
        # Render the result page, passing the prediction result to it.
        # results[0] is used to extract the single prediction value from the array.
        return render_template('result.html', results=results[0])

if __name__ == "__main__":
    # Run the Flask application in debug mode.
    # The host is set to '0.0.0.0' to make it accessible externally.
    app.run(host='0.0.0.0', port=5000, debug=True)

```

---

# End of Guide

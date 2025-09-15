End-to-End Machine Learning Project Guide
=========================================

This guide walks through a structured workflow for a machine learning regression project (e.g., predicting California house prices). We cover **environment setup**, **project structure**, **data exploration**, **exception handling**, **logging**, **data ingestion**, **data transformation**, **model training**, **prediction pipeline**, and **Flask app integration**. Each section includes clear steps, example code, and explanations. All code blocks have inline comments and are properly formatted for clarity.

Environment Setup
-----------------

*   conda create -n ml\_env python=3.9 -yconda activate ml\_env
    
*   **Project files:** In the project root, create:
    
    *   README.md – Overview of the project.
        
    *   requirements.txt – List of Python dependencies.
        
    *   setup.py – Installation and packaging script (if needed).
        
*   pandasnumpyseabornmatplotlibscikit-learncatboostxgboostdillflask-e .Here -e . allows installing the package in editable mode if you package this project.
    
*   from setuptools import find\_packages, setupfrom typing import Listdef get\_requirements(file\_path: str) -> List\[str\]: """Read requirements from a file and return them as a list, excluding editable installs.""" with open(file\_path) as f: requirements = f.readlines() requirements = \[req.strip() for req in requirements if req.strip()\] # Remove '-e .' if present requirements = \[req for req in requirements if req != '-e .'\] return requirementssetup( name='california\_house\_price', version='0.0.1', author='Mithu', author\_email='proshanta.mithu5@gmail.com', install\_requires=get\_requirements('requirements.txt'), packages=find\_packages(),)
    
    *   get\_requirements reads and cleans each line, removing any -e . entry.
        
    *   setup() uses find\_packages() to include all modules under src/.
        

Data Exploration
----------------

Before writing code, inspect your dataset in a Jupyter notebook or Python REPL:

*   **Understand data schema:** Use df.info() to see data types and non-null counts.
    
*   **Summary statistics:** df.describe() to view basic statistics (mean, median, etc.).
    
*   **Missing values:** df.isnull().sum() helps identify columns with null entries.
    
*   **Feature correlation:** Use df.corr() and visualizations (like heatmaps) to see relationships.
    
*   **Drop unnecessary columns:** If a column is irrelevant or duplicates others, drop it: df.drop(columns=\['col1', 'col2'\], inplace=True).
    
*   **Visualize data:** Plot distributions or scatterplots to understand target vs features.
    

This exploratory step helps identify which features are useful, potential preprocessing needs, and whether the problem is supervised (regression/classification) or unsupervised.

Project Structure
-----------------

Organize the project into clear directories. A typical structure for this pipeline is:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   project_root/  ├── src/  │   ├── components/  │   │   ├── __init__.py  │   │   ├── data_ingestion.py  │   │   ├── data_transformation.py  │   │   ├── model_trainer.py  │   ├── pipeline/  │   │   ├── prediction_pipeline.py  │   │   ├── train_pipeline.py  │   ├── templates/  │   │   ├── index.html  │   │   ├── predict_datapoint.html  │   │   ├── result.html  │   ├── __init__.py  │   ├── exception.py  │   ├── logger.py  │   └── utils.py  ├── artifacts/  │   ├── (models, preprocessor, data files, logs)  ├── requirements.txt  ├── setup.py  └── README.md   `

*   **src/components/**: Contains Python modules for each pipeline step (data ingestion, transformation, model training, etc.).
    
*   **src/pipeline/**: Contains pipeline scripts (e.g., scripts to run the whole training or prediction).
    
*   **src/templates/**: HTML files for a Flask web interface.
    
*   **src/exception.py** and **src/logger.py**: Utility modules for error handling and logging (described below).
    
*   **src/utils.py**: Reusable helper functions (e.g., saving/loading objects, model evaluation).
    
*   **artifacts/**: Stores output files like the raw dataset CSV, train/test splits, the saved preprocessor (.pkl), and trained model (.pkl or .sav).
    
*   **Virtual environment files** (requirements.txt, setup.py) and documentation (README.md) are in the project root.
    

Exception Handling (src/exception.py)
-------------------------------------

Define a custom exception to include context (file name and line number). This helps in debugging by showing where an error occurred.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import sys  from src.logger import logging  def error_message_details(error, error_details: sys) -> str:      """      Construct a detailed error message including the script name and line number.      """      _, _, exc_tb = error_details.exc_info()      file_name = exc_tb.tb_frame.f_code.co_filename      line_number = exc_tb.tb_lineno      error_message = (          f"Error occurred in script [{file_name}] "          f"at line [{line_number}]: {error}"      )      return error_message  class CustomException(Exception):      """Custom exception class that includes error details."""      def __init__(self, error_message, error_details=sys):          super().__init__(error_message)          # Store a formatted error message          self.error_message = error_message_details(error_message, error_details=error_details)      def __str__(self):          return self.error_message   `

*   **Usage:** In other modules, catch exceptions and raise CustomException(e, sys) to log detailed info.
    
*   try: # some operationexcept Exception as e: raise CustomException(e, sys)
    

Logging Setup (src/logger.py)
-----------------------------

Initialize logging to write detailed logs with timestamps. This helps trace execution flow and errors.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import logging  import os  from datetime import datetime  # Create a logs directory with current timestamp  LOG_DIR = os.path.join(os.getcwd(), "logs")  os.makedirs(LOG_DIR, exist_ok=True)  LOG_FILE = datetime.now().strftime('%Y_%m_%d_%H_%M_%S.log')  LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)  logging.basicConfig(      filename=LOG_FILE_PATH,      format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',      level=logging.INFO  )   `

*   **LOG\_DIR:** Directory for log files (logs/ in the project root).
    
*   **LOG\_FILE\_PATH:** Uses current date/time for unique log filenames.
    
*   **basicConfig:** Sets format (timestamp, line number, module, level, message) and log level.
    
*   **Usage:** Import logging in other modules (from src.logger import logging) and use logging.info(), logging.error(), etc.
    

Data Ingestion (src/components/data\_ingestion.py)
--------------------------------------------------

Read the raw dataset and split it into train and test sets. Store paths in a configuration dataclass.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML``   import os  import sys  import pandas as pd  from sklearn.model_selection import train_test_split  from src.logger import logging  from src.exception import CustomException  class DataIngestionConfig:      """Configuration for data ingestion paths."""      def __init__(self):          self.train_data_path = os.path.join('artifacts', 'train.csv')          self.test_data_path = os.path.join('artifacts', 'test.csv')          self.raw_data_path = os.path.join('artifacts', 'raw_data.csv')  class DataIngestion:      """Handles reading data and splitting into train/test."""      def __init__(self):          self.ingestion_config = DataIngestionConfig()      def initiate_data_ingestion(self, file_path: str):          """          Read the dataset from `file_path` and split into train/test CSV files.          Returns the paths to the train and test data files.          """          logging.info("Starting data ingestion process.")          try:              # Read the raw data              df = pd.read_csv(file_path)              logging.info("Dataset read into pandas DataFrame.")              # Ensure the artifacts directory exists              os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)              # Save the raw data for reference              df.to_csv(self.ingestion_config.raw_data_path, index=False)              logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}.")              # Split into training and testing sets              train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)              train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)              test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)              logging.info("Train and test data files created.")              return (                  self.ingestion_config.train_data_path,                  self.ingestion_config.test_data_path              )          except Exception as e:              logging.error("Exception occurred during data ingestion.")              raise CustomException(e, sys)   ``

*   **Usage:** Call DataIngestion().initiate\_data\_ingestion("path/to/your/data.csv") to process the dataset.
    
*   **Notes:** Ensure file\_path points to your raw CSV. The resulting train/test CSVs and a copy of raw data are saved under artifacts/.
    

Example usage (e.g., in a script or \_\_main\_\_ block):

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   if __name__ == "__main__":      try:          data_ingestion = DataIngestion()          train_path, test_path = data_ingestion.initiate_data_ingestion("data/california_housing.csv")          logging.info(f"Data ingestion completed. Train: {train_path}, Test: {test_path}")      except Exception as e:          logging.error("Data ingestion failed.")          raise CustomException(e, sys)   `

Data Transformation (src/components/data\_transformation.py)
------------------------------------------------------------

Create preprocessing pipelines to prepare data for modeling:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import os  import sys  import pandas as pd  import numpy as np  from sklearn.pipeline import Pipeline  from sklearn.preprocessing import StandardScaler, OrdinalEncoder  from sklearn.impute import SimpleImputer  from sklearn.compose import ColumnTransformer  from src.exception import CustomException  from src.logger import logging  from src.utils import save_object  class DataTransformationConfig:      """Configuration for data transformation artifact paths."""      def __init__(self):          self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')  class DataTransformation:      """Applies data preprocessing: imputation, encoding, scaling."""      def __init__(self):          self.config = DataTransformationConfig()      def get_data_transformer_object(self):          """          Create a preprocessor object (ColumnTransformer) that applies:            - Median imputation and StandardScaler to numeric features.            - Frequent-value imputation, ordinal encoding, and scaling to categorical features.          """          try:              logging.info("Initializing data transformation pipeline.")              # Define placeholder feature lists (replace with actual feature names)              numeric_features = ['feature1', 'feature2']        # e.g., ['MedInc', 'HouseAge']              categorical_features = ['feature3']               # e.g., ['ocean_proximity']              # Numeric pipeline: impute median, then scale              num_pipeline = Pipeline(steps=[                  ('imputer', SimpleImputer(strategy='median')),                  ('scaler', StandardScaler())              ])              # Categorical pipeline: impute mode, then ordinal encode, then scale              # Example: If 'feature3' has an ordered set of categories:              feature3_categories = ['category_low', 'category_medium', 'category_high']                cat_pipeline = Pipeline(steps=[                  ('imputer', SimpleImputer(strategy='most_frequent')),                  ('ordinal_encoder', OrdinalEncoder(categories=[feature3_categories])),                  ('scaler', StandardScaler())              ])              # Combine pipelines into a ColumnTransformer              preprocessor = ColumnTransformer(transformers=[                  ('num_pipeline', num_pipeline, numeric_features),                  ('cat_pipeline', cat_pipeline, categorical_features)              ])              logging.info("Preprocessing pipeline object created.")              return preprocessor          except Exception as e:              logging.error("Error creating data transformer object.")              raise CustomException(e, sys)      def initiate_data_transformation(self, train_path: str, test_path: str):          """          Fit the preprocessor on the training data and transform both train and test sets.          Returns transformed arrays and path to the saved preprocessor object.          """          try:              # Load datasets              train_df = pd.read_csv(train_path)              test_df = pd.read_csv(test_path)              logging.info("Train and test data loaded for transformation.")              # Specify the target column name              target_column = 'target'  # Replace with actual target column              # Drop target column from features              input_train_df = train_df.drop(columns=[target_column], errors='ignore')              target_train_df = train_df[target_column]              input_test_df = test_df.drop(columns=[target_column], errors='ignore')              target_test_df = test_df[target_column]              # Get the preprocessor pipeline and fit-transform on training data              preprocessing_obj = self.get_data_transformer_object()              input_train_arr = preprocessing_obj.fit_transform(input_train_df)              input_test_arr = preprocessing_obj.transform(input_test_df)              # Combine features and target back into arrays              train_arr = np.c_[input_train_arr, np.array(target_train_df)]              test_arr = np.c_[input_test_arr, np.array(target_test_df)]              # Save the preprocessor object for future use              save_object(file_path=self.config.preprocessor_obj_file_path,                          obj=preprocessing_obj)              logging.info(f"Preprocessor object saved at {self.config.preprocessor_obj_file_path}.")              return train_arr, test_arr, self.config.preprocessor_obj_file_path          except Exception as e:              logging.error("Error in initiating data transformation.")              raise CustomException(e, sys)   `

*   **Customize pipelines:** Replace the placeholder lists (numeric\_features, categorical\_features) with actual columns from your dataset.
    
*   **Ordinal categories:** Define ordered lists for any ordinal categorical columns if needed (as shown for feature3\_categories).
    
*   **Saving preprocessor:** The fitted preprocessor is saved using save\_object (defined in utils) so it can be used later during inference.
    

Utility Functions (src/utils.py)
--------------------------------

Include helper functions for saving/loading objects and evaluating models:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import os  import sys  import dill  import logging  from sklearn.model_selection import GridSearchCV  from sklearn.metrics import r2_score  from src.exception import CustomException  def save_object(file_path: str, obj):      """Save a Python object to a file using dill."""      try:          os.makedirs(os.path.dirname(file_path), exist_ok=True)          with open(file_path, 'wb') as file_obj:              dill.dump(obj, file_obj)          logging.info(f"Object saved to {file_path}")      except Exception as e:          raise CustomException(e, sys)  def load_object(file_path: str):      """Load a Python object from a file using dill."""      try:          with open(file_path, 'rb') as file_obj:              obj = dill.load(file_obj)          logging.info(f"Object loaded from {file_path}")          return obj      except Exception as e:          raise CustomException(e, sys)  def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):      """      Train and evaluate multiple models using hyperparameter tuning (GridSearchCV).      Returns a dictionary of test R^2 scores and a dictionary of best model instances.      """      try:          report = {}          best_models = {}          for model_name, model in models.items():              param_grid = params.get(model_name, {})              logging.info(f"Evaluating model: {model_name}")              if param_grid:                  grid_search = GridSearchCV(                      estimator=model, param_grid=param_grid,                      cv=3, n_jobs=-1, scoring='r2'                  )                  grid_search.fit(X_train, y_train)                  best_model = grid_search.best_estimator_                  logging.info(f"Best params for {model_name}: {grid_search.best_params_}")              else:                  # No hyperparameters provided, fit directly                  model.fit(X_train, y_train)                  best_model = model              best_models[model_name] = best_model              y_test_pred = best_model.predict(X_test)              test_score = r2_score(y_test, y_test_pred)              report[model_name] = test_score              logging.info(f"{model_name} test R2 score: {test_score:.4f}")          return report, best_models      except Exception as e:          raise CustomException(e, sys)   `

*   **save\_object / load\_object:** Use dill to serialize and deserialize Python objects (preprocessor, model). These functions ensure directories exist and handle exceptions.
    
*   **evaluate\_models:** Iterates through given models and hyperparameter grids (models and params are dictionaries keyed by model name). Uses GridSearchCV for tuning. Returns two dicts:
    
    *   One mapping model names to their test R² scores.
        
    *   Another mapping model names to the best (fitted) model instances.
        

Model Training (src/components/model\_trainer.py)
-------------------------------------------------

Train multiple regression models, tune hyperparameters, and select the best model.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import os  import sys  import numpy as np  from dataclasses import dataclass  from sklearn.linear_model import LinearRegression  from sklearn.tree import DecisionTreeRegressor  from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor  from xgboost import XGBRegressor  from catboost import CatBoostRegressor  from src.logger import logging  from src.exception import CustomException  from src.utils import save_object, evaluate_models  @dataclass  class ModelTrainerConfig:      """Configuration for model trainer."""      model_file_path: str = os.path.join('artifacts', 'model.pkl')  class ModelTrainer:      """Trains and selects the best regression model."""      def __init__(self):          self.config = ModelTrainerConfig()      def initiate_model_training(self, train_array, test_array):          """          Train multiple regression models and select the best by R2 score.          Returns the best model name and its score.          """          try:              logging.info("Starting model training.")              # Split arrays into features and target              X_train, y_train = train_array[:, :-1], train_array[:, -1]              X_test, y_test = test_array[:, :-1], test_array[:, -1]              # Define candidate models              models = {                  'Linear Regression': LinearRegression(),                  'Decision Tree': DecisionTreeRegressor(random_state=42),                  'Random Forest': RandomForestRegressor(random_state=42),                  'Gradient Boosting': GradientBoostingRegressor(random_state=42),                  'XGB Regressor': XGBRegressor(random_state=42),                  'CatBoost Regressor': CatBoostRegressor(verbose=False, random_state=42, train_dir=os.path.join(os.getcwd(), 'catboost_info')),                  'AdaBoost Regressor': AdaBoostRegressor(random_state=42)              }              # Hyperparameter grids for tuning              params = {                  'Decision Tree': {                      'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],                      'max_depth': [10, 20, None],                      'min_samples_split': [2, 5],                      'min_samples_leaf': [1, 2]                  },                  'Random Forest': {                      'n_estimators': [64, 128],                      'max_features': ['sqrt', 'log2', None],                      'max_depth': [10, 20, None],                      'min_samples_split': [2, 5],                      'min_samples_leaf': [1, 2]                  },                  'Gradient Boosting': {                      'learning_rate': [0.1, 0.01, 0.05],                      'subsample': [0.7, 0.8, 0.9],                      'n_estimators': [64, 128],                      'max_depth': [3, 5]                  },                  'XGB Regressor': {                      'learning_rate': [0.1, 0.01, 0.05],                      'subsample': [0.7, 0.9],                      'n_estimators': [64, 128],                      'max_depth': [3, 5],                      'colsample_bytree': [0.7, 0.9],                      'gamma': [0, 0.1]                  },                  'CatBoost Regressor': {                      'depth': [6, 8],                      'learning_rate': [0.01, 0.05],                      'iterations': [50, 100]                  },                  'AdaBoost Regressor': {                      'learning_rate': [0.1, 0.01, 0.5],                      'n_estimators': [64, 128]                  }              }              # Evaluate all models              model_report, best_models = evaluate_models(                  X_train=X_train, y_train=y_train,                  X_test=X_test, y_test=y_test,                  models=models, params=params              )              # Select the best model              best_model_name = max(model_report, key=model_report.get)              best_model_score = model_report[best_model_name]              best_model = best_models[best_model_name]              logging.info(f"Best model: {best_model_name} with R2 score {best_model_score:.4f}")              # Require minimum score (for example, 0.6)              if best_model_score < 0.6:                  raise CustomException(f"No model achieved R2 >= 0.6. Best was {best_model_score:.4f}", sys)              # Save the best model to a file              save_object(file_path=self.config.model_file_path, obj=best_model)              logging.info(f"Model saved at {self.config.model_file_path}.")              return best_model_name, best_model_score          except Exception as e:              logging.error("Exception in model training.")              raise CustomException(e, sys)   `

*   **Model dictionary:** Contains various regression models. Adjust or remove models as needed.
    
*   **Hyperparameters (params):** Tune these values as appropriate for your data. Grid ranges here are examples and may need expansion.
    
*   **evaluate\_models:** Returns test R² scores and the best estimator per model. We then pick the model with the highest R².
    
*   **Saving model:** The selected best model is saved to artifacts/model.pkl for later use.
    

Prediction Pipeline (src/pipeline/prediction\_pipeline.py)
----------------------------------------------------------

Load the trained model and preprocessor to make predictions on new data points:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import os  import pandas as pd  import logging  from src.utils import load_object  from src.exception import CustomException  class PredictPipeline:      """Loads model and preprocessor to predict on input features."""      def __init__(self):          self.model_path = os.path.join('artifacts', 'model.pkl')          self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')      def predict(self, features: pd.DataFrame):          """          Given a DataFrame of features, preprocess it and use the trained model to predict.          Returns the prediction array.          """          try:              # Load the saved model and preprocessor              model = load_object(file_path=self.model_path)              preprocessor = load_object(file_path=self.preprocessor_path)              # Apply preprocessing to the input features              data_scaled = preprocessor.transform(features)              logging.info("Data preprocessing completed.")              # Make predictions              prediction = model.predict(data_scaled)              logging.info("Prediction completed.")              return prediction          except Exception as e:              logging.error("Error in prediction pipeline.")              raise CustomException(e, sys)  class CustomData:      """      Structure for passing new data point features to the prediction pipeline.      """      def __init__(self, MedInc: float, HouseAge: float, AveRooms: float,                   AveBedrms: float, Population: float, AveOccup: float):          self.MedInc = MedInc          self.HouseAge = HouseAge          self.AveRooms = AveRooms          self.AveBedrms = AveBedrms          self.Population = Population          self.AveOccup = AveOccup      def get_data_as_dataframe(self) -> pd.DataFrame:          """          Convert the stored feature values into a pandas DataFrame (one row).          """          try:              input_data = {                  "MedInc": [self.MedInc],                  "HouseAge": [self.HouseAge],                  "AveRooms": [self.AveRooms],                  "AveBedrms": [self.AveBedrms],                  "Population": [self.Population],                  "AveOccup": [self.AveOccup]              }              return pd.DataFrame(input_data)          except Exception as e:              logging.error("Error converting custom data to DataFrame.")              raise CustomException(e, sys)   `

*   **PredictPipeline:** Loads the preprocessor and model from disk. The predict method takes a DataFrame of input features, applies transformations, and returns model predictions.
    
*   **CustomData:** Facilitates collecting input features (e.g., from a web form) and converting them into a DataFrame. Update or expand the constructor arguments to match all features your model expects.
    

Flask App Integration (app.py)
------------------------------

Create a simple Flask web app to serve the model. This app will have a homepage and a form to input data and display predictions.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from flask import Flask, request, render_template  import logging  from src.components.prediction_pipeline import PredictPipeline, CustomData  app = Flask(__name__)  @app.route('/')  def home():      """      Render the home page (index.html) with a button or link to start prediction.      """      return render_template('index.html')  @app.route('/predict_datapoint', methods=['GET', 'POST'])  def predict_datapoint():      """      GET: Render the input form (predict_datapoint.html).      POST: Collect form data, make prediction, and display the result.      """      if request.method == 'GET':          return render_template('predict_datapoint.html')      else:          # Collect input values from the form          data = CustomData(              MedInc=float(request.form.get('MedInc')),              HouseAge=float(request.form.get('HouseAge')),              AveRooms=float(request.form.get('AveRooms')),              AveBedrms=float(request.form.get('AveBedrms')),              Population=float(request.form.get('Population')),              AveOccup=float(request.form.get('AveOccup'))          )          # Convert inputs to DataFrame and make prediction          input_df = data.get_data_as_dataframe()          logging.info(f"Input data for prediction: {input_df.to_dict(orient='records')}")          predict_pipeline = PredictPipeline()          prediction = predict_pipeline.predict(input_df)          # Render the result page (result.html) with the prediction          return render_template('result.html', prediction=prediction[0])  if __name__ == "__main__":      app.run(host='0.0.0.0', port=5000, debug=True)   `

*   **Routes:**
    
    *   / – Home page (index.html). Provide navigation or info about the app.
        
    *   /predict\_datapoint – Handles both showing the form (GET) and processing submissions (POST). The HTML form (predict\_datapoint.html) should have input fields named MedInc, HouseAge, etc., matching request.form.get(...) keys.
        
*   **Templates:** You need corresponding HTML files in src/templates/:
    
    *   index.html – A simple welcome page with a link/button to the prediction form.
        
    *   predict\_datapoint.html – A form where users enter feature values. The form’s action should point to /predict\_datapoint and method="post".
        
    *   result.html – Displays the prediction result (passed as prediction context variable).
        
*   **Logging:** We log input data for debugging.
    

With the Flask app running, you can navigate to http://localhost:5000/ to access the UI, enter values, and view predictions.

This completes the end-to-end pipeline:

1.  **Setup** the environment and dependencies.
    
2.  **Explore** and understand your data.
    
3.  **Structure** the project for clarity.
    
4.  **Ingest** data and save splits.
    
5.  **Transform** data with preprocessing pipelines.
    
6.  **Train** and select the best model.
    
7.  **Deploy** a prediction pipeline and web app for inference.
    

Each code block above includes comments and should be placed in the specified files. Adjust placeholder feature/column names (feature1, target, etc.) to match your actual dataset. This structured guide serves as a clean, educational template for building a machine learning regression project.

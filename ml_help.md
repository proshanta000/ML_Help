##### 1. First of all have to make environment by 
```bash
conda create -p venv python==0.0(version) -y
``` 
##### Then create README.md, requirements.txt, setup.py files.
> README.md:
    `Information about projects`
> requirements.txt :
```text
            pandas
            numpy
            seaborn
            matplotlib
            scikit-learn
            catboost
            xgboost
            dill
            Flask

            -e.
```
`Also several models and framwork according to project type`

> Setup.py:
```python
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT ='-e'


def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements =file_obj.readline()
        [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

# Version and details 
setup(
    name= 'California House Price',
    version= '0.0.1',
    author= 'Mithu',
    author_email= 'proshanta.mithu5@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages= find_packages(),
)
```

`Also requird any thing else if needed`

##### 2. At first have to check the dataset for features  and and target variables. BY jupyter notebook , have to check  data types supearvised lor un supearvised dataset,   info(df.info()),  descriptions (df.describe()), nall value (df.isnull().sum()), features not nessery can find using use corelations (df.corr()), and can drop(df.drop([....], axis=1)), and exprement many other things.

>  After that have to make piplines folders like :
```bash
project_root/
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/
│   │   ├── __init__.py 
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

##### 4. First have to write exception.py

> Excption.py:
```python
import sys
from src.logger import logging

def error_message_details(error, error_details:sys):
    _, _, exc_tb = error_details.exc_info()
    file_name =exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    # Constracting a details error message
    error_message = "Error occured in python script name [{0}] Line number [{1}] error message [{2}]".format(
        file_name, line_number, str(error)
    )

    return error_message



class CustomException(Exception):
    def __init__(self, error_message, error_details=sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details= error_details)
    
    def __str__(self):
        return self.error_message
```

##### 5. 2nd have to create logger.py
> logger.py:
```python
import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "Logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s -%(levelname)s -%(message)s",
    level=logging.INFO
)
```
##### 6. 3rd have to start  writting  Data_ingestion 
> Import libraries :
```python
import os
import sys

# Get the absolute path of the current script's directory.
# This ensures that no matter where you run the script from,
# it can always find the project's root directory.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up two levels to reach the project root directory
# For example: E:/Data_science/Projects/california_house_price
project_root = os.path.join(script_dir, '..', '..')

# Add the project root to the system path
# This allows Python to find and import modules from the 'src' directory
sys.path.append(project_root)

# Now, your imports will work correctly.
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# If any thing else needed can add here 
```
> Then write code about initiate data ingestion:
```python
## Intialize the Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    # Configuration for data ingestion paths
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'raw_data.csv')


class DataIngestion:
    # Initializing the data ingestion configaration
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    #Initiating the data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")

        try:
            df = (....) #Import dataset
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

        
            logging.info("Train Teat and split.")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is compleated")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e, sys)
```
> Now the code to check the ingestion part:
```python
if __name__=="__main__":
# The main execution block of the pipeline
 try:
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

except Exception as e:
    logging.info("An exception occurred during the pipeline execution.")
    raise CustomException(e, sys)
# It will change according to file completion 
```
##### 7. Then have to start writing data transformation
> Import Libraries :
```python
import os 
import sys
import pandas as pd 
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
```
>  Then write code about data transformation:
```python 
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated.")
            
            # The columns to apply standardization on
            # These must be the columns that exist in the DataFrame after dropping the target and other columns.
            features_to_scale = [...] #Features name
            ## Pipeline for scaling
            scaler_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )



            # If there is Numeracial and catagorical issus 
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = [....]
            numerical_cols = [...]
            
            # If there is ordinal variables 
            # Define the custom ranking for each ordinal variable
            ..._categories = [....]  # Name should be according to columns name and variables according unique data
            ...._categories = [....] # Name should be according to columns name and variables according unique data
            
            # Here only two exemple but can be added as far as needed

            logging.info('Pipeline Initiated')


            # pipline line exemple 

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')), # If median needed
                ('scaler',StandardScaler()) # if standardscaler needed
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')), # Where simple imputer needed
                ('ordinalencoder',OrdinalEncoder(categories=[...._categories, ..._categories])), # Where ordinal encoder needed and catagoris name should be according previous name 
                ('scaler',StandardScaler()) # if standardscaler needed
                ]

            )
            
            # Preprocessor part 
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols), # (name, transformer, columns)/ ('name of the pipline', 'name of the pipline', 'name of the columns')
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])


            

            # Create a ColumnTransformer to apply the scaling pipeline
            preprocessor = ColumnTransformer([
                ('scaler_pipeline', scaler_pipeline, features_to_scale)
            ])
            
            logging.info('Pipeline for data transformation completed.')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data transformation.")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete.")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object.")
            preprocessing_object = self.get_data_transformation_object()

            target_column_name = "..." #Target column name 
            # Drop columns that are not features for the model
            drop_column_name = [target_column_name, '...'] #column want to drope

            input_feature_train_df = train_df.drop(columns=drop_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test data.")

            # Transform the feature data
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            # Combine transformed features with the target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


                #> Have to go utils.py and create save_object
                           # > save object will be  [
                                1. Important libraries :
                                    [
                                        import os 
                                        import sys
                                        import pickle
                                        import numpy as np
                                        import pandas as pd
                                        import dill

                                        from src.exception import CustomException
                                        from src.logger import logging
                                    ]
                                2. Save object:
                                    [
                                        def save_object(file_path, obj):
                                            try:
                                                dir_path = os.path.dirname(file_path)
                                                os.makedirs(dir_path, exist_ok=True)

                                                with open(file_path, 'wb')as file_obj:
                                                    pickle.dump(obj, file_obj)

                                            except Exception as e:
                                                raise CustomException(e, sys)
                                    ]
            ### Now have to add the sorce in data_transformation.py > import libraries section 
            from src.utils import save_object

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            logging.info('Preprocessor pickle file saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Exception occurred in the initiate_data_transformation.')
            raise CustomException(e, sys)
```
>  Now the code to check write inside data ingestion  to check data transformation part:
```python
if __name__=="__main__":
    # The main execution block of the pipeline
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        

    except Exception as e:
        logging.info("An exception occurred during the pipeline execution.")
        raise CustomException(e, sys)
# It will change according to file completion 
```

##### 8. Now have to start writing model trainer file(model_trainer.py):
> import important libraries:
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
from src.utils import save_object
```
> Then start Model traning code writing:
```python
@dataclass
class ModelTraningConfig:
    """
    Configuration for model training path and parameters
    """
    traning_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    """
    Class for training machine learning models
    """
    def __init__(self):
        self.model_traning_config = ModelTraningConfig()
    
    def initiate_model_traning(self, train_array, test_array):
        """
        This method trains multiple regression models and evaluates their performance.
        It returns the best model based on r2 score.
        """
        try:
            logging.info("Entering the model training method or component.")

            # Fix for the CatBoostError: create an absolute path for the writable directory
            catboost_dir = os.path.join(os.getcwd(), 'artifacts', 'catboost_info')
            os.makedirs(catboost_dir, exist_ok=True)
            logging.info(f"Created directory for CatBoost: {catboost_dir}")

            logging.info("Splitting the training and testing input and target variables.")
            # Splitting the train and test arrays into features and target variables
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            # Here all is for suparvised data
            #Creating models dictionary with various regression models
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
                'XGB Regressor': XGBRegressor(random_state=42),
                'KNeighbors Regressor': KNeighborsRegressor(),
                # Fix for the CatBoostError: specify a writable directory
                'CatBoost Regressor': CatBoostRegressor(verbose=False, random_state=42, train_dir=catboost_dir),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42)
            }

            # Define parameters grid for each model
            # IMPORTANT: Adjust these parameters and ranges based on dataset and computational resources.
            # Start with broader ranges and narrow down as results come.
            params = {
                'Linear Regression': {}, # No hyperparameters for basic Linear Regression in this context
                # Corrected key name to match the models dictionary
                'Decision Tree Regressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [10, 20, None], # Adjusted range for faster execution
                    'min_samples_split': [2, 5], # Adjusted range
                    'min_samples_leaf': [1, 2] # Adjusted range
                },
                'Random Forest Regressor': {
                    'n_estimators': [64, 128, 256], # Adjusted range
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [10, 20, None], # Adjusted range
                    'min_samples_split': [2, 5], # Adjusted range
                    'min_samples_leaf': [1, 2] # Adjusted range
                },
                'Gradient Boosting Regressor': {
                    'learning_rate': [.1, .01, .05], # Adjusted range
                    'subsample': [0.7, 0.8, 0.9], # Adjusted range
                    'n_estimators': [64, 128, 256], # Adjusted range
                    'max_depth': [3, 5] # Adjusted range
                },
                'XGB Regressor': {
                    'learning_rate': [.1, .01, .05], # Adjusted range
                    'n_estimators': [64, 128, 256], # Adjusted range
                    'max_depth': [3, 5], # Adjusted range
                    'subsample': [0.7, 0.8], # Adjusted range
                    'colsample_bytree': [0.7, 0.8], # Adjusted range
                    'gamma': [0, 0.1] # Adjusted range
                },
                'KNeighbors Regressor': {
                    'n_neighbors': [5, 7, 9], # Adjusted range
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree'] # Adjusted range
                },
                'CatBoost Regressor': {
                    'depth': [6, 8], # Adjusted range
                    'learning_rate': [0.01, 0.05], # Adjusted range
                    'iterations': [50, 100] # Adjusted range
                },
                'AdaBoost Regressor': {
                    'learning_rate': [.1, .01, 0.5], # Adjusted range
                    'n_estimators': [64, 128, 256] # Adjusted range
                }
            }

            # Creating models dictionary with various classification models
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
                'Random Forest Classifier': RandomForestClassifier(random_state=42),
                'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
                'XGB Classifier': XGBClassifier(random_state=42),
                'KNeighbors Classifier': KNeighborsClassifier(),
                'CatBoost Classifier': CatBoostClassifier(verbose=False, random_state=42),
                'AdaBoost Classifier': AdaBoostClassifier(random_state=42)
            }

            # Define parameters grid for each model
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
            

            # Here for unsupervised data 

            # Creating models dictionary with various unsupervised models
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
            
            > have to create evalute_models on utils:
                evalute_models:
                    [ 
                        def evalute_models(X_train, y_train, X_test, y_test, models, params):
                            """
                            Evaluates multiple machine learning models with hyperparameter tuning.

                            Args:
                                X_train (np.array): Training features.
                                y_train (np.array): Training target.
                                X_test (np.array): Testing features.
                                y_test (np.array): Testing target.
                                models (dict): A dictionary of model instances (e.g., {'Linear Regression': LinearRegression()}).
                                params (dict): A dictionary of hyperparameter grids/distributions for each model,
                                            keyed by model name.

                            Returns:
                                tuple: A tuple containing:
                                    - dict: A dictionary containing the R2 score for each best-tuned model on the test set.
                            """

                            try:
                                report = {}
                                tuned_models={} # To store the best model instant after tuning

                                for model_name, model in models.items():
                                    param_grid = params.get(model_name, {}) # Get Parameters for the current model, default to empty dict

                                    logging.info(f"starting evaluting/tuning for {model_name}...")

                                    if param_grid: # if the parameters are defined , perform tuning
                                        logging.info(f"performimg gridsearchcv for {model_name} with parameters: {param_grid}")
                                        # Use GridSearchCV for hyperparameter tuning
                                        grid_search = GridSearchCV(
                                            estimator=model,
                                            param_grid=param_grid,
                                            cv=3, # Number of cross-validation folds. Adjust based on dataset size and computational resources.
                                            n_jobs=-1, # Use all available CPU cores for parallel processing
                                            verbose=0, # Set to 1 or 2 for more detailed output during search
                                            scoring='r2' # Use R2 score as the evaluation metric for regression
                                        )

                                        grid_search.fit(X_train, y_train)

                                        best_model = grid_search.best_estimator_
                                        best_params = grid_search.best_params_
                                        logging.info(f"Best parameters for {model_name}: {best_params}")
                                        logging.info(f"Best cross-validation R2 score for {model_name}: {grid_search.best_score_:.4f}")
                                    else:
                                        logging.info(f"No specific hyperparameters defined for {model_name}. Training with default parameters.")
                                        model.fit(X_train, y_train)
                                        best_model = model # Use the default model if no params for tuning

                                    # Store the best tuned model (or default model if no tuning was done)
                                    tuned_models[model_name] = best_model

                                    # Predict on the training and testing data using the best model
                                    y_train_pred = best_model.predict(X_train)
                                    y_test_pred = best_model.predict(X_test)

                                    # Calculate R2 scores
                                    train_score = r2_score(y_train, y_train_pred)
                                    test_score = r2_score(y_test, y_test_pred)

                                    # Store the model name and its R2 score (test score) in the report dictionary
                                    report[model_name] = test_score

                                    logging.info(f"{model_name} - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

                                return report, tuned_models # Return both the scores and the tuned models

                            except Exception as e:
                                logging.info("")
                                raise CustomException(e, sys)
                    ]

                    2. ### Now have to add the sorce in model_trainer.py > import libraries section 
                      from src.utils import evalute_modelsbject 

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
> Now the code to check write inside data ingestion  to check model trainer part:
```python
if __name__=="__main__":
    # The main execution block of the pipeline
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)


        model_trainer = ModelTrainer()
        # Capture the returned values from the model trainer
        best_model_name, best_model_score, best_model = model_trainer.initiate_model_traning(train_arr, test_arr)
        
        # Now we can print the actual results
        print(f"\nModel Training Complete.")
        print(f"Best Model Found: {best_model_name}")
        print(f"R2 Score: {best_model_score:.4f}\n")

    except Exception as e:
        logging.info("An exception occurred during the pipeline execution.")
        raise CustomException(e, sys)
# It will change according to file completion 
```


##### 9. Now have to start writing predict pipline(predict_pipline.py):
> import important libraries:
```python
import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
```
> Then have to start predict pipline code:
```python
class PreddictPipline:
    """
    class for making prediction  using a trained model
    """
    def __init__(self):
        """self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)
        """
        pass

        > Have to create load object on utils:
            1. load_object:
                [
                    def load_object(file_path):
                        """
                        Load an object from a file using pickle.
                        
                        Args:
                            file_path (str): Path to the file from which the object is to be loaded.
                        
                        Returns:
                            obj: The loaded object.
                        """
                        try:
                            with open(file_path, 'rb') as file_obj:
                                obj = dill.load(file_obj)
                            logging.info(f"Object loaded from {file_path}")
                            return obj
                        except Exception as e:
                            raise CustomException(e, sys)
               ]
            2. ### Now have to add the sorce in predict_pipline.py > import libraries section 
                      from src.utils import load_object 
    
    def predict(self, feature):
        model_path = 'artifacts/model.pkl'
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)

        try:
            #preprocess the input features
            data_scaled= preprocessor.transform(feature)
            logging.info('Data Preprocessing conpleted')

            # Make prediction using the model

            prediction = model.predict(data_scaled)
            logging.info("prediction completed")

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    """
    Class for handling custom data input for prediction .
    """
    def __init__(self,
                    MedInc : float,
                    HouseAge : float # This is demo Have to write according the dataset 
                    ):

            self.MedInc = MedInc
            self.HouseAge = HouseAge
            # This is demo Have to write according the dataset 
        
    def get_data_as_dataframe(self):
         """
        Converts the custom data instance into a pandas DataFrame.
            
        This is a crucial step as most machine learning models are trained
        to accept input in a DataFrame format.
            
        Returns:
            pd.DataFrame: A DataFrame containing the custom data.
        """
        try:
            custom_data_input_dict = {
                    "MedInc": [self.MedInc],
                    "HouseAge": [self.HouseAge],
                    # This is demo Have to write according the dataset 
                }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
                # Raise a custom exception if there is an error in data conversion
                logging.error("Error occurred while creating DataFrame from custom data.")
                raise CustomException(e, sys)

```
##### 10. Have to start about model trainer , It will be add later 
> start writing model app.py:
> important libraries:
```python
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import PreddictPipline, CustomData
```

> Now start writing app data:
```python
application = Flask(__name__)

app = application
# Route for the home page
@app.route('/')  
def home():
    """
    Renders the main input form page (index.html).
    """
    return render_template('index.html')

# Have to write code in index.htmal for predect page:
    <a href="{{ url_for('predict_datapoint') }}" class="inline-block w-full py-4 px-6 bg-blue-600 text-white font-bold text-lg rounded-full shadow-lg hover:bg-blue-700 transition duration-300 transform hover:scale-105">
    Start Your Prediction
</a>

# Route to handle both displaying the form and processing the prediction
@app.route('/predict_datapoint', methods=['GET', 'POST']) # Have to write code in predict_datapoint.html (action="{{ url_for('predict_datapoint')}}" method="post")
def predict_datapoint():
    """
    Handles displaying the prediction form (GET) and processing
    the form submission (POST) to make a prediction.
    """
    if request.method == 'GET':
        # If the request is a GET, render the form page
        return render_template('predict_datapoint.html')
    else:
        # If the request is a POST (form submission), process the data
        # Create a CustomData object from form inputs there is some example
        data = CustomData(
            MedInc=float(request.form.get('MedInc')),
            HouseAge=float(request.form.get('HouseAge'))
        )
        
        # Convert CustomData to a DataFrame for the prediction pipeline
        pred_df = data.get_data_as_dataframe()
        print(pred_df) # For debugging

        # Initialize and run your prediction pipeline
        predict_pipeline = PreddictPipline()
        results = predict_pipeline.predict(pred_df)
        
        # Render the result page, passing the prediction result
        return render_template('result.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
```


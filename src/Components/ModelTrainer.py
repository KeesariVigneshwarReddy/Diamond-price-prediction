import os
import sys
import copy
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from src.Exception import CustomException
from src.Logger import logging

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

@dataclass 
class ModelTrainerConfig :
    
    X_test_scaled_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_test_scaled.csv")

    report_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Training results", "report.csv")
    params_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Training results", "params.csv")

class ModelTrainer :
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model_regression(self, X_test, true, predicted) :
        mse = mean_squared_error(true, predicted)
        mae = mean_absolute_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        adj_r2_square = 1 - (1 - r2_square) * (len(true) - 1)/(len(true) - X_test.shape[1] - 1)
        return mse, mae, rmse, r2_square, adj_r2_square 
    

    def initate_model_training(self, X_train_data_path, y_train_data_path, X_test_data_path, y_test_data_path, numerical_scaler_obj_file_path) :
        logging.info("Entered the Model Trainer component")
        try :
            # Load train test datasets
            X_train = pd.read_csv(X_train_data_path)
            y_train = pd.read_csv(y_train_data_path)
            X_test = pd.read_csv(X_test_data_path)
            y_test = pd.read_csv(y_test_data_path)
            logging.info("X_train, y_train, X_test, y_test are loaded")

            # Scale X_test 
            numerical_scaler = joblib.load(numerical_scaler_obj_file_path)
            X_test_scaled = numerical_scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_train.columns.tolist())
            X_test_scaled.to_csv(self.model_trainer_config.X_test_scaled_data_path, index = False, header = True)
            X_test = X_test_scaled
            logging.info("X_test is scaled")

            # models
            models ={
                        "Linear Regression": LinearRegression(),
                    }
            logging.info("Models initialized")

            # params 
            params ={
                        "Linear Regression" : {}
                     }
            logging.info("Parameters initialized")

            # model training and evaluating
            report = {}
            model_params = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                gs = GridSearchCV(model, para, cv = 3, scoring = "r2")
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                
                values = self.evaluate_model_regression(X_test, y_test, y_test_pred)
                report[list(models.keys())[i]] = {var: val for var, val in zip(('mse', 'mae', 'rmse', 'r2_square', 'adj_r2_square'), values)}
                model_params[list(models.keys())[i]] = dict(gs.best_params_)
            logging.info("All the models are trained with respective parameters.")

            # Report generating
            report_df = pd.DataFrame.from_dict(report, orient = 'index')
            model_params_df = pd.DataFrame.from_dict(model_params, orient = 'index')
            report_df.to_csv(self.model_trainer_config.report_data_path)
            model_params_df.to_csv(self.model_trainer_config.params_data_path)
            logging.info("Report and params is generated")

            # Best model with Best score
            best_model_score = 0
            for key1, value1 in report.items() :
                if value1['adj_r2_square'] > best_model_score :
                    best_model_score = max(best_model_score, value1['adj_r2_square'])
                    best_model = key1

            logging.info("Model training completed")
            return (
                        self.model_trainer_config.X_test_scaled_data_path,
                        self.model_trainer_config.report_data_path, 
                        self.model_trainer_config.params_data_path,
                        best_model,
                        best_model_score
                    )
        
        except Exception as e:
            logging.info('Exception occured in initiate_model_training')
            raise CustomException(e, sys)
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
from src.Exception import CustomException

@dataclass 
class BestModelConfig:
   best_model_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Machine Learning models", "best_model.pkl")

class BestModel :
    def __init__(self) :
        self.best_model_config = BestModelConfig()
    
    def get_best_model_path(self, X_train_data_path, y_train_data_path) :
        logging.info("Implementing the best model")
        try :
            # Load train test datasets
            X_train = pd.read_csv(X_train_data_path)
            y_train = pd.read_csv(y_train_data_path)
            logging.info("X_train, y_train datasets are loaded")

            # Train best model
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            joblib.dump(reg, self.best_model_config.best_model_obj_file_path)
            logging.info("Best model is implemented and saved")

            return (self.best_model_config.best_model_obj_file_path)
        
        except Exception as e:
            logging.info('Exception occured at get_best_model_path')
            raise CustomException(e, sys)
        
if __name__ == '__main__' :
    X_train_data_path = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_train_scaled.csv")
    y_train_data_path = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "y_train.csv")
    best_model = BestModel()
    best_model_obj_file_path = best_model.get_best_model_path(X_train_data_path, y_train_data_path)
    print("Best model is implemented and saved")
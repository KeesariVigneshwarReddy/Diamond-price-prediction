import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LinearRegression
from src.Exception import CustomException
from src.Logger import logging

from dataclasses import dataclass
import sys
import os

@dataclass 
class BestModelConfig:
   best_model_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Machine Learning models", "best_model.pkl")

class BestModel :
    def __init__(self) :
        self.best_model_config = BestModelConfig()
    
    def get_best_model_path(self, X_train_scaled_encoded_data_path, y_train_data_path) :
        logging.info("Implementing the best model")
        try :
            # Load train test datasets
            X_train = pd.read_csv(X_train_scaled_encoded_data_path)
            y_train = pd.read_csv(y_train_data_path)
            
            logging.info("Train datasets are loaded")

            reg = LinearRegression()
            reg.fit(X_train, y_train)
            joblib.dump(reg, self.best_model_config.best_model_obj_file_path)

            logging.info("Best model is implemented ans saved")
            return (self.best_model_config.best_model_obj_file_path)
        
        except Exception as e:
            logging.info('Exception occured at Developing best model')
            raise CustomException(e, sys)
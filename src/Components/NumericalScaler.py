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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class NumericalScalerConfig :
    
    X_train_scaled_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_train_scaled.csv")

    numerical_scaler_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Transform models", "numerical_encoder.pkl")

class NumericalScaler :
    def __init__(self):
        self.numerical_scaler_config = NumericalScalerConfig()

    def initiate_numerical_scaler(self, X_train_data_path) :
        logging.info("Entered the Numerical Scaler component")
        try :
            # Load X_train
            X_train = pd.read_csv(X_train_data_path)
            logging.info('Read X_train as dataframe')

            # Numerical scaling - standard scaling
            num_scaler_pipeline = Pipeline(steps = [
                                                ('std_scaler', StandardScaler())
                                                #('min_max_scaler', MinMaxScaler())
                                                   ]
                                          )
            """ The unit vector scaling
            X_train_normalized = normalize(X_train)
            X_train_scaled = pd.DataFrame(X_train_normalized, columns = X_train.columns.tolist())
            """
            numerical_scaler = ColumnTransformer([
                                                        ('num_pipeline', num_scaler_pipeline, X_train.columns.tolist())
                                                ])
            
            X_train_scaled = pd.DataFrame(numerical_scaler.fit_transform(X_train), columns = numerical_scaler.get_feature_names_out())
            X_train_scaled.columns = [col.split('_')[-1] for col in X_train_scaled.columns]

            joblib.dump(numerical_scaler, self.numerical_scaler_config.numerical_scaler_obj_file_path)

            X_train_scaled.to_csv(self.numerical_scaler_config.X_train_scaled_data_path, index = False, header = True)

            logging.info("Numerical columns of X_train dataset is scaled")

            return (
                        self.numerical_scaler_config.X_train_scaled_data_path,
                        self.numerical_scaler_config.numerical_scaler_obj_file_path
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_numerical_scaler")

            raise CustomException(e, sys)
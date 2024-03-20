import os
import sys
import copy
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr, skew, kurtosis
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, normalize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.Exception import CustomException
from src.Logger import logging
import os

from dataclasses import dataclass

@dataclass
class ResamplerConfig :
    
    X_resampled_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "06_X_resampled.csv")
    y_resampled_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "07_X_resampled.csv")

class Resampler :
    def __init__(self):
        self.resampler_config = ResamplerConfig()

    def initiate_resampler(self, X_imputed_data_path, y_data_path) :
        logging.info("Entered the data transformation component")
        try :
            
            # Load X_imputed and y
            X = pd.read_csv(X_imputed_data_path)
            y = pd.read_csv(y_data_path)
            logging.info('Read X_imputed and y as dataframe')

            # Upsample the data set
            # Determine this from EDA
            threshold = 0.2 # Determine this from EDA
            sampling_strategy_ratio = {5 : 1, 6 : 0.9368575624082232, 7 : 0.2922173274596182, # Classes which should not be upsampled 
                                4 : 0.3, 8 : 0.5, 3 : 0.6} # Classes to be upsampled write expected ratio
            
            highest_frequency_class = y.value_counts().idxmax()
            highest_frequency_count = y.value_counts().max()

            imbalance_ratios = {}
            for class_label, count in y.value_counts().items():
                imbalance_ratios[class_label] =  count / highest_frequency_count

            classes_tobe_upsampled = []
            for key, value in imbalance_ratios.items() :
                if imbalance_ratios.get(key) <= threshold :
                    classes_tobe_upsampled.append(key)

            # Detrmine the sampling strategy and apply SMOTE
            sampling_strategy = {}
            for key, value in sampling_strategy_ratio.items() :
                sampling_strategy[key] = int(highest_frequency_count * value)

            Sm = SMOTE(sampling_strategy = sampling_strategy, random_state = 42)

            X_resampled, y_resampled = Sm.fit_resample(X, y)

            X_resampled.to_csv(self.resampler_config.X_resampled_data_path, index = False, header = True)
            y_resampled.to_csv(self.resampler_config.y_resampled_data_path, index = False, header = True)

            logging.info('Resampler has done its job')

            return (
                        self.resampler_config.X_resampled_data_path,
                        self.resampler_config.y_resampled_y_data_path
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_resampler")

            raise CustomException(e,sys)
        
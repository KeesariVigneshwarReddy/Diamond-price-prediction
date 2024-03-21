import os
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
import sys
from dataclasses import dataclass

@dataclass
class DataTransformationConfig :
    
    X_imputed_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "05_X_imputed.csv")
    X_resampled_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "06_X_resampled.csv")
    y_resampled_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "07_X_resampled.csv")
    X_num_target_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "08_X_num_target.csv")
    X_num_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "09_X_num.csv")
    X_num_removed_low_variance_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "10_X_num_removed_low_variance.csv")
    X_MIF_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "11_X_MIF.csv")
    X_train_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "12_X_train.csv")
    y_train_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "13_y_train.csv")
    X_test_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "14_X_test.csv")
    y_test_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "15_y_test.csv")
    X_train_scaled_encoded_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "16_X_train_scaled_encoded.csv")

    imputer_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Imputer models", "imputer.pkl")
    scaler_encoder_obj_file_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Models/Transform models", "scaler_encoder.pkl")

class DataTransformation :
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_imputer_object(self, num_columns, cat_columns, num_columns_strategy) :
        try :
            num_imputer_pipeline = Pipeline(steps = [
                                                        ('imputer', ColumnTransformer([
                                                                                        ('impute' + col, SimpleImputer(strategy = strategy), [col]) for col, strategy in num_columns_strategy.items()
                                                                                    ])
                                                        )
                                                        # (imputer, SimpleImputer(strategy = 'mean'))
                                                    ]
                                            )

            cat_imputer_pipeline = Pipeline(steps = [
                                                        ('imputer', SimpleImputer(strategy = 'most_frequent')),
                                                    ]
                                            )

            imputer = ColumnTransformer([
                                            ('num_imputer_pipeline', num_imputer_pipeline, num_columns), 
                                            ('cat_imputer_pipeline', cat_imputer_pipeline, cat_columns)
                                        ])
            return imputer
        
        except Exception as e :

            logging.info("Exception occured in the creating imputer object")
            raise CustomException(e,sys)
        
    def get_feature_scaler_encoder_object(self, num_columns, cat_columns, cat_nominal_columns, cat_ordinal_columns, cat_target_ordinal_columns, cut_cat, color_cat, clarity_cat) :
        try :
            
            return feature_scaler_encoder
        
        except Exception as e :

            logging.info("Exception occured in the creating feature scaler encoder object")
            raise CustomException(e,sys)

    def resample_data_up(self, X, y, threshold, sampling_strategy_ratio : dict) :
        try :
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

            Sm = SMOTE(sampling_strategy = sampling_strategy, random_state=42)

            X_resampled, y_resampled = Sm.fit_resample(X, y)

            return (X_resampled, y_resampled)
        
        except Exception as e :

            logging.info("Exception occured in Upsampling data")
            raise CustomException(e,sys)

    def correlation(self, X_num_target, X_num, threshold_min, threshold_max) :
        try :
            spearman_corr_matrix_target, p_values = spearmanr(X_num_target)
            spearman_corr_df_target = pd.DataFrame(spearman_corr_matrix_target, columns = X_num_target.columns, index = X_num_target.columns)
            spearman_corr_matrix, p_values = spearmanr(X_num)
            spearman_corr_df = pd.DataFrame(spearman_corr_matrix, columns = X_num.columns, index = X_num.columns)

            # Removing the features with are least correlated with target
            features_not_correlated_with_target = [] 

            for i in range(len(spearman_corr_df_target.columns)) :
                if abs(spearman_corr_df_target.iloc[i, len(spearman_corr_df_target.columns) - 1]) < threshold_min :
                    features_not_correlated_with_target.append(spearman_corr_df_target.columns[i])

            # Removing the features with are highly correlated to each other
            threshold = 0.85 

            features_correlated = {} 

            for i in range(len(spearman_corr_df.columns)) :
                for j in range(len(spearman_corr_df.columns)) :
                    if i == j :
                        continue
                    if spearman_corr_df.iloc[i, j] > threshold_max :
                        list_key = features_correlated.get(spearman_corr_df.columns[i], [])
                        list_key.append(spearman_corr_df.index[j])
                        features_correlated[spearman_corr_df.columns[i]] = list_key

            return (features_not_correlated_with_target, features_correlated)
        except Exception as e :

            logging.info("Exception occured in Correlation of numeric features function")
            raise CustomException(e,sys)

    def initiate_data_transformation(self, X_data_path, y_data_path) :
        logging.info("Entered the data transformation component")
        try :
            
            # Load X and y
            X = pd.read_csv(X_data_path)
            y = pd.read_csv(y_data_path)
            logging.info('Read X and y as dataframe')

            # Features division
            num_columns = X.select_dtypes(include = 'number').columns.tolist()
            cat_columns = X.select_dtypes(include = 'object').columns.tolist()
            cat_nominal_columns = []
            cat_ordinal_columns = ['cut', 'color', 'clarity']
            cat_target_ordinal_columns = []
            cut_cat = ['Fair', 'Good', 'Very Good','Ideal', 'Premium']
            color_cat = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
            clarity_cat = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            logging.info('Features division conmpleted')

            # handling missing values
            """
            num_columns_strategy = {
                                        'carat' : 'mean',
                                        'depth' : 'median',
                                        'table' : 'mean',
                                        'x' : 'mean',
                                        'y' : 'median',
                                        'z' : 'mean'
                                    }
            imputer = self.get_imputer_object(num_columns, cat_columns, num_columns_strategy)
            joblib.dump(imputer, self.data_transformation_config.imputer_obj_file_path)
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns = imputer.get_feature_names_out())
            X_imputed.columns = [col.split('_')[-1] for col in X_imputed.columns]
            X_imputed.to_csv(self.data_transformation_config.X_imputed_data_path, index = False, header = True)
            X = X_imputed
            logging.info("Missing values are handled")
            """

            # Upsample the data set
            """
            # Determine this from EDA
            threshold = 0.2 # Determine this from EDA
            sampling_strategy_ratio = {5 : 1, 6 : 0.9368575624082232, 7 : 0.2922173274596182, # Classes which should not be upsampled 
                                4 : 0.3, 8 : 0.5, 3 : 0.6} # Classes to be upsampled write expected ratio
            X_resampled, y_resampled = self.resample_data_up(X, y, threshold, sampling_strategy_ratio)
            X_resampled.to_csv(self.data_transformation_config.X_resampled_data_path, index = False, header = True)
            y_resampled.to_csv(self.data_transformation_config.y_resampled_data_path, index = False, header = True)

            X = X_resampled
            y = y_resampled
            logging.info("Resampling of data set is done.")
            """

            # Feature engineering
            # Isolating numerical features
            X_num_target = pd.concat([X[num_columns], y], axis = 1)
            X_num = pd.DataFrame(X[num_columns])
            X_num_target.to_csv(self.data_transformation_config.X_num_target_data_path, index = False, header = True)
            X_num.to_csv(self.data_transformation_config.X_num_data_path, index = False, header = True)
            logging.info("Numerical features are isolated")

            # Drop low variance features - Determine threshold according to your use case
            """
            var_thres_selector = VarianceThreshold(threshold = 0.0000001)
            var_thres_selector.fit_transform(X_num)
            constant_columns = [column for column in X_num.columns
                                if column not in X_num.columns[var_thres_selector.get_support()]]
            X_num = X_num.drop(constant_columns, axis = 1)
            X_num.to_csv(self.data_transformation_config.X_num_removed_low_variance_data_path, index = False, header = True)
            logging.info("Low variance columns are removed")
            """
            # Correlation of features on the basis of spearman correlation coefficient
            threshold_min = 0.1 # min correlation needed to be selected
            threshold_max = 0.85 # max correlation needed to be selected
            features_not_correlated_with_target, features_correlated = self.correlation(X_num_target, X_num, threshold_min, threshold_max)

            features_to_be_dropped = copy.deepcopy(features_not_correlated_with_target) + ['x', 'y', 'z'] # Features to be dropped  
                                                                                                          # if they are highly correlated. Select from EDA or features_correlated        
            for i in features_to_be_dropped :
                num_columns.remove(i)
            X.drop(columns = features_to_be_dropped, inplace = True)
            X.to_csv(self.data_transformation_config.X_MIF_data_path, index = False, header = True)
            logging.info("Features are removed on the basis of correlation")

            # Train test split
            # Split ratio train : test
            split_ratio = [80, 20]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = float(split_ratio[1] / 100), random_state = 42)
            X_train.reset_index(drop = True, inplace = True)
            y_train.reset_index(drop = True, inplace = True)
            X_test.reset_index(drop = True, inplace = True)
            y_test.reset_index(drop = True, inplace = True)
            X_train.to_csv(self.data_transformation_config.X_train_data_path, index = False, header = True)
            y_train.to_csv(self.data_transformation_config.y_train_data_path, index = False, header = True)
            X_test.to_csv(self.data_transformation_config.X_test_data_path, index = False, header = True)
            y_test.to_csv(self.data_transformation_config.y_test_data_path, index = False, header = True)
            logging.info('Train test split completed')

            # Numerical scaling and categorical encoding
            num_scaler_pipeline = Pipeline(steps = [
                                                ('std_scaler', StandardScaler())
                                                #('min_max_scaler', MinMaxScaler())
                                                   ]
                                          )
            cat_ordinal_encoder_pipeline = Pipeline(steps = [
                                                                ('ordinalencoder', OrdinalEncoder(categories = [cut_cat, color_cat, clarity_cat])), # order should be same as in dataframe
                                                                ('scaler', StandardScaler())
                                                                # target guided encoding
                                                            ]
                                                    )
            feature_scaler_encoder = ColumnTransformer([
                                                            ('num_pipeline', num_scaler_pipeline, num_columns), 
                                                            #('cat_nominal_pipeline', cat_nominal_encoder_pipeline, cat_nominal_columns),
                                                            ('cat_ordinal_pipeline', cat_ordinal_encoder_pipeline, cat_ordinal_columns)  
                                                    ])
            #feature_scaler_encoder = self.get_feature_scaler_encoder_object(num_columns, cat_columns, cat_ordinal_columns, cat_nominal_columns, cat_target_ordinal_columns, cut_cat, color_cat, clarity_cat)
            joblib.dump(feature_scaler_encoder, self.data_transformation_config.scaler_encoder_obj_file_path)
            X_train_scaled_encoded = pd.DataFrame(feature_scaler_encoder.fit_transform(X_train), columns = feature_scaler_encoder.get_feature_names_out())
            X_train_scaled_encoded.columns = [col.split('_')[-1] for col in X_train_scaled_encoded.columns]

            # X_train_scaled_encoded = pd.concat([X_train[num_columns], X_train_scaled_encoded], axis = 1)

            X_train_scaled_encoded.to_csv(self.data_transformation_config.X_train_scaled_encoded_data_path, index = False, header = True)
            logging.info("X_train dataset is scaled as well as encoded")


            logging.info('Data transformation completed')

            return (
                        self.data_transformation_config.X_train_scaled_encoded_data_path,
                        self.data_transformation_config.y_train_data_path,
                        self.data_transformation_config.X_test_data_path,
                        self.data_transformation_config.y_test_data_path,
                        self.data_transformation_config.imputer_obj_file_path,
                        self.data_transformation_config.scaler_encoder_obj_file_path
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
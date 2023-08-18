import os
import sys
from dataclasses import dataclass
import numpy as np
from src.utils import evaluate_model, save_object, store_training_cols
from imblearn.over_sampling import SMOTE
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig():
    model_file_path = os.path.join("Artifacts","model.pkl")

class ModelTrainer():
    def __init__(self, train_arr,test_arr):
        self.model_trainer_config = ModelTrainerConfig()
        self.train_arr = train_arr
        self.test_arr = test_arr

    def run(self):
        X_train,X_test,Y_train,Y_test = self.oversampling()
        X_train,X_test,Y_train,Y_test = self.feature_selection(X_train,X_test,Y_train,Y_test)
        self.initiate_model_training(X_train,X_test, Y_train,Y_test)

    
    def oversampling(self):
        """
        Method Name: oversampling
        Description: Oversamples the minority class using the SMOTE method to deal with the imbalanced nature of data
        Output: Returns the resampled X and Y dataframes
        """
        try:
            X_train, Y_train, X_test, Y_test = self.train_arr[:-1], self.train_arr[-1], self.test_arr[:-1], self.test_arr[-1]
            smote = SMOTE()
            X_resampled, Y_resampled = smote.fit_resample(X_train,Y_train)
            return X_resampled, X_test, Y_resampled, Y_test
        except Exception as e:
            raise CustomException(e,sys)





    def feature_selection(self,X_train, X_test,Y_train,Y_test):
        """
        Method Name: feature_selection
        Description: Checks for collinearity among predictors and removes highly correlated features
        Output: Returns 4 dataframes i.e X_train, X_test, Y_train,Y_test after removing correlated features
        """
        try:
            corr_matrix = X_train.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
            X_train.drop(to_drop, axis=1, inplace=True)
            X_test.drop(to_drop, axis=1,inplace=True)
            logging.info("Multicollinearity in data has been dealt with")
            training_cols = X_train.columns
            store_training_cols(training_cols)
            return X_train,X_test,Y_train,Y_test
        
        except Exception as e:
            raise CustomException(e,sys)




     def initiate_model_training(self,X_train,X_test,Y_train,Y_test):
        """
        Method Name: initiate_model_training
        Description:Trains the different models on the training data and performs model selection on test data.
        Output: Does not return anything but stores the best performing model in the relevant directory.
        """
        models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),}
        params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'n_estimators': [ 20, 40, 80, 100],'max_depth': [ 5, 10, 20],'min_samples_split': [2, 4, 8, 12]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
            
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

        report =evaluate_model(X_train,Y_train,X_test,Y_test,models,params)
            
            ## To get best model score from dict
        best_model_score = max(sorted(report.values()))

            ## To get best model name from dict

        best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
        
        best_model = models[best_model_name]
        #if best_model_score<0.6:
            #raise CustomException("No best model found")
            #logging.info(f"Best found model on both training and testing dataset")

            #save_object(
                #file_path=self.model_trainer_config.trained_model_file_path,
                #obj=best_model
            #)

        predicted=best_model.predict(X_test)
        predicted = predicted.astype('int64')

        accuracy = accuracy_score(Y_test, predicted)
        return accuracy
    

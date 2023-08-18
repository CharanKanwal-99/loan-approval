import dill
import json
import numpy as np
from sklearn.metrics import r2_score,accuracy_score

def save_object(object,file_path):
    with open(file_path) as file:
        dill.dump(object,file)

def object_loader(file_path):
    with open(file_path) as file:
        object = dill.load(file)
    return object

def find_redundant(df):
    percent_missing = (df.isna().sum()/len(df))*100
    to_drop = []
    for key,val in percent_missing.items():
        if val > 20:
            to_drop.append(key)
    return to_drop

def store_training_cols(cols):
    schema_cols = {k: np.nan for k in cols}
    with open('schema_cols.json','wb') as file:
        json.dump(schema_cols, file)
    
    
def evaluate_model(X_train, Y_train, X_test,Y_test,models,param):
    try:
        report = {}


        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)



            pred = model.predict(X_test)
            pred = pred.astype('int64')

            test_model_score = accuracy_score(Y_test, pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

from src.utils import object_loader

class PredictionPipeline():
    
    def __init__(self):
        pass

    def predict(self,features):
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model_path = 'artifacts/model.pkl'
        model = object_loader(model_path)
        preprocessor_obj = object_loader(preprocessor_path)
        scaled_features = preprocessor_obj.transform(features)
        prediction = model.predict(scaled_features)
        return prediction

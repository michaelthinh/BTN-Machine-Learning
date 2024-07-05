from model.lstm_model import LSTMModelPredictService, LSTMModel, LSTMModelBuilder
from model.rnn_model import RNNModelPredictService, RNNModel, RNNModelBuilder
from model.xgboost_model import XGBModelPredictService, XGBModel, XGBModelBuilder

class ModelPredictServiceFactory:
    @staticmethod
    def getModelPredictService(modelName, features, coin):
        if modelName == "LSTM":
            return LSTMModelPredictService(model=LSTMModel(features=features, coin=coin))
        elif modelName == "RNN":
            return RNNModelPredictService(model=RNNModel(features=features, coin=coin))
        elif modelName == "XGB":
            return XGBModelPredictService(model=XGBModel(features=features, coin=coin))
        else:
            raise Exception("Model not found")

class ModelBuilderFactory:
    @staticmethod
    def getModelBuilder(modelName, features, coin):
        if modelName == "LSTM":
            return LSTMModelBuilder(model=LSTMModel(features=features, coin=coin))
        elif modelName == "RNN":
            return RNNModelBuilder(model=RNNModel(features=features, coin=coin))
        elif modelName == "XGB":
            return XGBModelBuilder(model=XGBModel(features=features, coin=coin))
        else:
            raise Exception("Model not found")
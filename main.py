from model.init import ModelsInitializer
from model.train_data import TrainDataProvider
from model.lstm_model import LSTMModelPredictService, LSTMModel
from model.rnn_model import RNNModelPredictService, RNNModel
from model.xgboost_model import XGBModelPredictService, XGBModel
from constant import coins, windowSize, candel_columns, features
from model.factory import ModelPredictServiceFactory

# from sklearn.metrics import mean_squared_error

# ModelsInitializer().init()
features = features
coin = coins[0]
trainDataProvider = TrainDataProvider(
    coin=coin, features=features, windowSize=windowSize
)
lstmModelPredictService = LSTMModelPredictService(
    model=LSTMModel(features=features, coin=coin)
)
rnnModelPredictService = RNNModelPredictService(
    model=RNNModel(features=features, coin=coin)
)
xgbModelPredictService = XGBModelPredictService(
    model=XGBModel(features=features, coin=coin)
)

lstmFromFactory = ModelPredictServiceFactory.getModelPredictService(
    "LSTM", features, coin
)
rnnFromFactory = ModelPredictServiceFactory.getModelPredictService(
    "RNN", features, coin
)
xgbFromFactory = ModelPredictServiceFactory.getModelPredictService(
    "XGB", features, coin
)


def getSample():
    data = trainDataProvider.getDataFromFile()
    input = data[features][:windowSize]
    output = data[candel_columns][windowSize : windowSize + 1]
    return input, output


sampleInput, sampleOutput = getSample()
print("actual", sampleOutput)
lstmlPredictedOutput = lstmModelPredictService.execute(sampleInput)
rnnPredictedOutput = rnnModelPredictService.execute(sampleInput)
xgbPredictedOutput = xgbModelPredictService.execute(sampleInput[-1:])
print("lstm", lstmlPredictedOutput)
print("rnn", rnnPredictedOutput)
print("xgb", xgbPredictedOutput)
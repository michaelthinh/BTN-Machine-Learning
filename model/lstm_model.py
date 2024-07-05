from model.base import (
    SavedModelPredictService,
    Model,
    ModelBuilder,
    WindowedModelInputValidator,
    KerasModelFileService,
)
from model.train_data import TrainDataProvider
from constant import windowSize, lstm_units, candel_columns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from model.loader import KerasModelLoader
import pandas as pd


class LSTMModel(Model):
    def __init__(self, features, coin):
        super().__init__("LSTM", features, coin)


class LSTMModelBuilder(ModelBuilder):
    def __init__(self, model: Model):
        lstmModel = LSTMModel(model.features, model.coin)
        super().__init__(
            model=lstmModel, modelFileService=KerasModelFileService(model=lstmModel)
        )
        self.dataProvider = TrainDataProvider(
            coin=model.coin, features=model.features, windowSize=windowSize
        )

    def buildModel(self):
        x_train, y_train = self.dataProvider.getTrainData()
        inputShape = (x_train.shape[1], x_train.shape[2])

        model = Sequential()
        model.add(Input(shape=inputShape))
        model.add(
            LSTM(
                units=lstm_units,
                return_sequences=True,
            )
        )
        model.add(LSTM(units=lstm_units))
        model.add(Dense(units=y_train.shape[1]))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
        model.save(self.modelFileService.getModelFileName())


class LSTMModelPredictService(SavedModelPredictService):
    def __init__(self, model: LSTMModel):
        super().__init__(
            model=model,
            modelLoader=KerasModelLoader(model),
            inputValidator=WindowedModelInputValidator(model),
        )

    def predictWithLoadedModel(self, loaded_model, data: pd.DataFrame) -> pd.DataFrame:
        data = data.values.reshape(1, data.shape[0], data.shape[1])
        prediction = loaded_model.predict(data)
        df_prediction = pd.DataFrame(prediction, columns=candel_columns)
        return df_prediction

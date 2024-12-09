from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, AdamW
from keras_tuner.tuners import RandomSearch

class NeuralNetwork:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.model = init_model()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def init_model(self):
        '''
        Defines the architecture of the MLP

        Returns:
            model: keras model that can be compiled
        '''
        model = Sequential([
        Dense(64, input_dim=self.X_train.shape[1], activation='leaky_relu'),
        Dropout(0.2),
        Dense(64, activation='leaky_relu'),
        Dropout(0.2),
        Dense(128, activation='leaky_relu'),
        Dropout(0.2),
        Dense(128, activation='leaky_relu'),
        Dropout(0.2),
        Dense(64, activation='leaky_relu'),
        Dropout(0.2),
        Dense(16, activation='leaky_relu'),
        Dropout(0.2),
        Dense(1)  # Output layer
        ])
        return model

    def compile_model(self):
        '''
        Compiles the model for training
        '''
        self.model = model.compile(optimizer=AdamW(learning_rate=0.001), loss='huber', metrics=['mae'])

    def train(self):
        '''
        Fits the model to training data
        '''
        self.model.fit(
        self.X_train, self.y_train,
        validation_data=(self.X_val, self.y_val),
        epochs=250,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        )

    def test(self):
        '''
        Tests the model and saves the learning-curve in a figure.

        Returns:
            score, plot
        '''
        test_loss, test_mae = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('learning-curve.png')
        return score, plt

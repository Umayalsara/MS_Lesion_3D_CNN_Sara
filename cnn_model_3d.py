import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras_tuner import HyperModel, HyperParameters

class CNN3DHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Conv3D(filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
                         kernel_size=(3, 3, 3), activation='relu', input_shape=(32, 32, 32, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
                         kernel_size=(3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        model.add(Flatten())
        model.add(Dense(units=hp.Int('units', min_value=64, max_value=256, step=64), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

if __name__ == "__main__":
    # Example usage
    input_shape = (32, 32, 32, 1)  # Example input shape
    hypermodel = CNN3DHyperModel()
    
    # Create a HyperParameters object
    hp = HyperParameters()
    
    # Use the HyperParameters object to build the model
    model = hypermodel.build(hp=hp)  # Pass the HyperParameters object to the build method
    model.summary()

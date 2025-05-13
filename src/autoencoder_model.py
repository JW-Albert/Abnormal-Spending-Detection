import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def get_autoencoder_scores(X, y, encoding_dim=8, epochs=50, batch_size=16):
    # 只用正常樣本訓練
    X_train = X[y == 1]
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    recon = autoencoder.predict(X)
    recon_error = np.mean((X - recon)**2, axis=1)
    return recon_error 
import pickle
import numpy as np

from gensim.models import Word2Vec

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense
)
from tensorflow.keras.optimizers import Adam

# =========================
# SETTINGS
# =========================

VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
LSTM_UNITS = 64
EPOCHS = 10
BATCH_SIZE = 16

# =========================
# GPU CHECK
# =========================

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# =========================
# LOAD DATA
# =========================

with open("encoder_input.pkl", "rb") as f:
    encoder_input_data = pickle.load(f)

with open("decoder_input.pkl", "rb") as f:
    decoder_input_data = pickle.load(f)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# LOAD WORD2VEC
# =========================

w2v_model = Word2Vec.load("word2vec.model")

# =========================
# EMBEDDING MATRIX
# =========================

embedding_matrix = np.zeros(
    (VOCAB_SIZE, EMBEDDING_DIM)
)

for word, idx in tokenizer.word_index.items():

    if idx >= VOCAB_SIZE:
        continue

    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]

# =========================
# DECODER TARGET
# =========================

decoder_target_data = np.zeros_like(
    decoder_input_data
)

decoder_target_data[:, :-1] = \
    decoder_input_data[:, 1:]

# =========================
# MODEL
# =========================

encoder_inputs = Input(
    shape=(encoder_input_data.shape[1],)
)

embedding_layer = Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    trainable=True
)

encoder_embedding = embedding_layer(
    encoder_inputs
)

encoder_lstm = LSTM(
    LSTM_UNITS,
    return_state=True
)

_, state_h, state_c = encoder_lstm(
    encoder_embedding
)

encoder_states = [state_h, state_c]

# =========================
# DECODER
# =========================

decoder_inputs = Input(
    shape=(decoder_input_data.shape[1],)
)

decoder_embedding = embedding_layer(
    decoder_inputs
)

decoder_lstm = LSTM(
    LSTM_UNITS,
    return_sequences=True,
    return_state=True
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)

decoder_dense = Dense(
    VOCAB_SIZE,
    activation="softmax"
)

decoder_outputs = decoder_dense(
    decoder_outputs
)

# =========================
# BUILD MODEL
# =========================

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

# =========================
# COMPILE
# =========================

model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================

model.fit(
    [encoder_input_data, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1
)

# =========================
# SAVE MODEL
# =========================

model.save("chatbot.h5")

print("\n================================")
print("Training Complete")
print("Model Saved : chatbot.h5")
print("================================")
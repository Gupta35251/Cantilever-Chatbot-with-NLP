from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 10

model = load_model("chatbot.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}

def generate_response(user_text):

    seq = tokenizer.texts_to_sequences(
        [user_text.lower()]
    )

    seq = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    start_token = tokenizer.word_index.get(
        "start",
        1
    )

    decoder_input = np.zeros(
        (1, MAX_LEN)
    )

    decoder_input[0, 0] = start_token

    response_words = []

    for i in range(MAX_LEN - 1):

        prediction = model.predict(
            [seq, decoder_input],
            verbose=0
        )

        predicted_id = np.argmax(
            prediction[0, i]
        )

        word = index_to_word.get(
            predicted_id,
            ""
        )

        if word == "end":
            break

        if word not in [
            "start",
            "end",
            "<OOV>",
            ""
        ]:
            response_words.append(word)

        if i + 1 < MAX_LEN:
            decoder_input[0, i + 1] = predicted_id

    if len(response_words) == 0:
        return "Sorry, I don't understand."

    return " ".join(response_words)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.form["message"]

    bot_reply = generate_response(
        user_message
    )

    return jsonify({
        "response": bot_reply
    })

if __name__ == "__main__":
    app.run(debug=True)
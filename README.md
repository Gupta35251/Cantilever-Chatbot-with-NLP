# Cantilever-Chatbot-with-NLP

# Chatbot Using Word2Vec, Seq2Seq LSTM and Flask

## Project Overview

This project implements a chatbot using Natural Language Processing (NLP) techniques. The chatbot is trained on the Cornell Movie Dialog Dataset and generates responses based on user input.

The project uses:

* Word2Vec for word embeddings
* Seq2Seq (Sequence-to-Sequence) architecture
* LSTM (Long Short-Term Memory) networks
* Flask for the web interface

---

## Objectives

* Collect and preprocess conversational data.
* Convert text into numerical representations using Word2Vec.
* Train a Seq2Seq LSTM model to generate responses.
* Build a web-based chatbot interface using Flask.
* Deploy the chatbot on a cloud platform.

---

## Dataset

### Cornell Movie Dialog Corpus

The dataset contains movie conversations between characters.

Dataset includes:

* Movie dialogues
* Question-answer pairs
* Conversational text data

---

## Technologies Used

* Python
* TensorFlow / Keras
* Gensim
* Flask
* NumPy
* Pandas
* NLTK

---

## Project Workflow

1. Dataset Collection
2. Text Cleaning and Preprocessing
3. Tokenization
4. Word2Vec Embedding Generation
5. Seq2Seq LSTM Model Training
6. Response Generation
7. Flask Web Application
8. Deployment

---

## Project Structure

chatbot_project/

├── data/

│ ├── movie_lines.txt

│ └── movie_conversations.txt

├── preprocess.py

├── train_word2vec.py

├── train_chatbot.py

├── chatbot.py

├── app.py

├── templates/

│ └── index.html

├── tokenizer.pkl

├── encoder_input.pkl

├── decoder_input.pkl

├── max_len.pkl

├── word2vec.model

├── chatbot.h5

├── requirements.txt

└── Procfile

---

## Model Architecture

User Input

↓

Word2Vec Embeddings

↓

Encoder LSTM

↓

Context Vector

↓

Decoder LSTM

↓

Generated Response

---

## Preprocessing Steps

* Convert text to lowercase
* Remove special characters
* Create question-answer pairs
* Add start and end tokens
* Tokenize text
* Pad sequences

---

## Training Configuration

MAX_SAMPLES = 5000

VOCAB_SIZE = 3000

MAX_LEN = 10

EMBEDDING_DIM = 50

LSTM_UNITS = 64

BATCH_SIZE = 16

EPOCHS = 10

---

## How to Run

### Step 1

Run preprocessing:

python preprocess.py

### Step 2

Train Word2Vec:

python train_word2vec.py

### Step 3

Train chatbot:

python train_chatbot.py

### Step 4

Start Flask application:

python app.py

### Step 5

Open browser:

http://127.0.0.1:5000

---

## Features

* Conversational chatbot
* Word embedding generation
* Sequence-to-sequence learning
* LSTM-based response generation
* Web-based user interface
* Easy deployment

---

## Future Enhancements

* Attention Mechanism
* Transformer-based chatbot
* Hugging Face Embeddings
* Better conversational datasets
* Cloud deployment on Heroku or Render

---

## Conclusion

This project demonstrates how Natural Language Processing techniques such as Word2Vec, Seq2Seq, and LSTM can be used to build a conversational chatbot. The chatbot learns from conversational datasets and generates responses through sequence prediction while providing a user-friendly interface through Flask.

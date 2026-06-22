import re
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# SETTINGS
# =========================
MAX_SAMPLES = 5000
VOCAB_SIZE = 3000
MAX_LEN = 10

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", text)
    text = text.strip()
    return text

# =========================
# LOAD CORNELL DATASET
# =========================
lines = {}

with open(
    "data/movie_lines.txt",
    encoding="iso-8859-1",
    errors="ignore"
) as f:

    for line in f:
        parts = line.split(" +++$+++ ")

        if len(parts) == 5:
            line_id = parts[0]
            text = parts[4].strip()

            lines[line_id] = clean_text(text)

# =========================
# CREATE QUESTION-ANSWER PAIRS
# =========================
questions = []
answers = []

count = 0

with open(
    "data/movie_conversations.txt",
    encoding="iso-8859-1",
    errors="ignore"
) as f:

    for line in f:

        if count >= MAX_SAMPLES:
            break

        parts = line.split(" +++$+++ ")

        if len(parts) < 4:
            continue

        conversation = parts[-1]

        ids = re.findall(r"L[0-9]+", conversation)

        for i in range(len(ids) - 1):

            q = lines.get(ids[i], "")
            a = lines.get(ids[i + 1], "")

            if len(q) > 0 and len(a) > 0:

                questions.append(q)
                answers.append(a)

                count += 1

                if count >= MAX_SAMPLES:
                    break

# =========================
# ADD START END TOKENS
# =========================
answers = [
    "<start> " + a + " <end>"
    for a in answers
]

# =========================
# TOKENIZER
# =========================
tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    questions + answers
)

# =========================
# TEXT TO SEQUENCES
# =========================
encoder_sequences = tokenizer.texts_to_sequences(
    questions
)

decoder_sequences = tokenizer.texts_to_sequences(
    answers
)

# =========================
# PAD SEQUENCES
# =========================
encoder_input_data = pad_sequences(
    encoder_sequences,
    maxlen=MAX_LEN,
    padding="post"
)

decoder_input_data = pad_sequences(
    decoder_sequences,
    maxlen=MAX_LEN,
    padding="post"
)

# =========================
# SAVE FILES
# =========================
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("encoder_input.pkl", "wb") as f:
    pickle.dump(encoder_input_data, f)

with open("decoder_input.pkl", "wb") as f:
    pickle.dump(decoder_input_data, f)

with open("max_len.pkl", "wb") as f:
    pickle.dump(MAX_LEN, f)

print("=================================")
print("Preprocessing Complete")
print("Question-Answer Pairs:", len(questions))
print("Vocabulary Size:", len(tokenizer.word_index))
print("Encoder Shape:", encoder_input_data.shape)
print("Decoder Shape:", decoder_input_data.shape)
print("=================================")
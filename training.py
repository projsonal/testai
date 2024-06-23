import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional

# Load dataset
data = pd.read_csv('a-backup.csv', sep='|')

# Preprocessing function to clean text
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()

# Apply preprocessing to questions and answers
data['question'] = data['question'].apply(preprocess_text)
data['answer'] = data['answer'].apply(preprocess_text)

# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['question'].tolist() + data['answer'].tolist())
question_sequences = tokenizer.texts_to_sequences(data['question'].tolist())
answer_sequences = tokenizer.texts_to_sequences(data['answer'].tolist())

max_len = max([len(seq) for seq in question_sequences + answer_sequences])
question_sequences = pad_sequences(question_sequences, maxlen=max_len, padding='post')
answer_sequences = pad_sequences(answer_sequences, maxlen=max_len, padding='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(question_sequences, answer_sequences, test_size=0.2)


# Build Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

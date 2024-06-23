from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Define tokenizer and max_len globally
tokenizer = Tokenizer()
max_len = 0  # Initialize max_len

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global max_len  # Access max_len globally
    user_input = request.form['user_input']
    response = generate_response(user_input, tokenizer, max_len)
    return jsonify({'response': response})

def generate_response(user_input, tokenizer, max_len):
    # Convert user input to sequence
    sequence = tokenizer.texts_to_sequences([user_input])
    sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Predict response
    prediction = app.model.predict(sequence)  # Access model from app
    response_sequence = tf.argmax(prediction[0], axis=-1).numpy()
    
    # Convert sequence to text
    response = tokenizer.sequences_to_texts([response_sequence])
    return response[0]

if __name__ == '__main__':
    # Load and compile model
    # Contoh app.model = load_model
    app.model = load_model('model.h5')  # Load your trained model
    app.run(debug=True)

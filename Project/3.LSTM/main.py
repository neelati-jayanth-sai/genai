import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
st.title('Next Word Prediction (LSTM)')

model_lstm = load_model('hamlet_model(LSTM).h5')
tokenizer_lstm = pickle.load(open('tokenizer.pickle', 'rb'))

# Create a function to predict the next word for LSTM
def generate_text_lstm(seed_text, model, max_sequence_len):
    token_list = tokenizer_lstm.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer_lstm.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Create a text input box for LSTM
input_text_lstm = st.text_input('Enter a sentence or phrase (LSTM):', key='input_lstm')
max_sequence_len_lstm = model_lstm.input_shape[1] + 1

# Create a button to generate the next word for LSTM
if st.button('Generate Next Word (LSTM)', key='button_lstm'):
    if input_text_lstm:
        next_word_lstm = generate_text_lstm(input_text_lstm, model_lstm, max_sequence_len_lstm)
        st.write(f'The next word is: {next_word_lstm}')
    else:
        st.write('Please enter a sentence or phrase to generate the next word.')

# Load the GRU model
st.title('Next Word Prediction (GRU)')

model_gru = load_model('hamlet_model(GRU).h5')
tokenizer_gru = pickle.load(open('tokenizer.pickle', 'rb'))

# Create a function to predict the next word for GRU
def generate_text_gru(seed_text, model, max_sequence_len):
    token_list = tokenizer_gru.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer_gru.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Create a text input box for GRU
input_text_gru = st.text_input('Enter a sentence or phrase (GRU):', key='input_gru')
max_sequence_len_gru = model_gru.input_shape[1] + 1

# Create a button to generate the next word for GRU
if st.button('Generate Next Word (GRU)', key='button_gru'):
    if input_text_gru:
        next_word_gru = generate_text_gru(input_text_gru, model_gru, max_sequence_len_gru)
        st.write(f'The next word is: {next_word_gru}')
    else:
        st.write('Please enter a sentence or phrase to generate the next word.')

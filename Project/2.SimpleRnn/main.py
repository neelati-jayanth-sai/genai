import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras.api.datasets import imdb
from keras.api.preprocessing import sequence
from keras.api.models import load_model

#load imdb data set

word_index = imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}



#Loading model
model=load_model('simple_rnn_imdb.h5')



def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

def preprocessing(text):
    words=text.lower().split(" ")
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocessing(review)
    prediction=model.predict(preprocessed_input)
    sentiment="Positive" if prediction>0.5 else "Negative"

    return sentiment, prediction[0][0]

import streamlit as st

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below:")

user_input=st.text_area("Movie review")

if st.button("Predict"):
    preprocess_input=preprocessing(user_input)
    prediction=model.predict(preprocess_input)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"
    st.write("Sentiment: ", sentiment)
    st.write("Prediction: ", prediction[0][0])
else:
    st.write("Enter a movie review to predict sentiment")
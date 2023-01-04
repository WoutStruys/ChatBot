#!/usr/bin/env python3

import nltk
import emoji
import pickle
import numpy as np
import streamlit as st
from nltk.corpus import stopwords

class Chatbot:
    def __init__(self):
        self.model = pickle.load(open("model.pkl","rb"))
        self.tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
        nltk.download('stopwords')
        self.emoji_dict = { 0 : "❤️‍🩹", 1 : "⚾", 2 : "😄", 3 : "😞", 4 : "🍴"}
        self.stop_words = set(stopwords.words("english"))
        
    def predict_emoji(self, text: str):
        filtered_texts = []
        #We halen de engelse stopwoorden uit de nltk library
        #We filteren deze uit onze tweets, deze zeggen namelijk niets over het sentiment
        words = text.split() 
        words = [word for word in words if word not in self.stop_words]
        filtered_text = " ".join(words)
        filtered_texts.append(filtered_text)
        print("Filtered text: " + str(filtered_texts))
        tokenized_text = self.tokenizer.texts_to_sequences(filtered_texts)
        predict_x = self.model.predict(tokenized_text)
        pred = np.argmax(predict_x, axis=1)
        if pred.size <= 0:
            return None
        print(pred[0])
        return pred[0]
    
    def get_emoji(self, text):
        response = self.predict_emoji(text)
        return emoji.emojize(self.emoji_dict[response])
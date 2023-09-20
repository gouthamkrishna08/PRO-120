#Text Data Preprocessing Lib
import nltk

import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

def bot_response(user_input):
    
    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
   
    # Input data encoding 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
    prediction=model.predict(np.array(bag))
    predicted_label=np.argmax(prediction[0])
    predicted_class=classes[predicted_label]
    for x in intents["intents"]:
        if x["tag"]==predicted_class:
            bot_response=random.choice(x["responses"])
            return bot_response
 

while True:
    user_input=input("Type Your message here :  ")
    response=bot_response(user_input)
    print(response) 
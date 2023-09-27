import streamlit as st
import numpy as np
import random
import string
import nltk
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
import pickle
import json
from itertools import count
from PIL import Image
import base64
import io



#Import the Image class from PIL (Pillow)

# Opening JSON file
f = open('intent.json',)
key_counter = count()

# Returns JSON object as a dictionary
data = json.load(f)
stemmer = PorterStemmer()

stop_words = ['is', 'am', 'the', 'a', 'an', 'be', 'are', 'were']

words = []
y = []
patterns = []
x = []
classes = []

# Getting the classes(tags) output and patterns from 'data'
for intent in data['intents']:
    classes.append(intent['tag'])
    for pattern in intent['patterns']:
        # Applying tokenizer to convert the sentences into a list of words
        tokens = nltk.word_tokenize(pattern)
        # Here .extend() is used instead of append() since we don't want to append lists in words
        # but its elements, i.e., words in 'token'
        # Here 'tokens' is a list of words
        words.extend(tokens)
        patterns.append(pattern)
        y.append(intent['tag'])

# Converting to lowercase, applying lemmatization, and removing the punctuations
# Here 'words' is our vocabulary containing all the words
words = [stemmer.stem(word.lower()) for word in words if word not in string.punctuation and
         word not in stop_words]
# Converting the list to a set to avoid doubling of words in 'words'
words = sorted(set(words))
words = list(words)

for list_ in patterns:
    list_ = nltk.word_tokenize(list_)
    list_ = [stemmer.stem(lis.lower()) for lis in list_ if lis not in string.punctuation and lis not in stop_words]
    x.append(list_)

# Load the Random Forest model
with open("random_forest_model.sav", 'rb') as model_file:
    loaded_job_model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.sav", 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load the chatbot
chatbot = load_model('mymodel.h5')

# Initialize a counter for generating unique keys
key_counter = count()

# Define job details prediction function
def job_details_prediction(QualLevel, CGPA, SourceId, CurrentExperience):
    # Extract the features for the new data point
    new_structured_features = np.array([[CGPA, QualLevel, CurrentExperience, SourceId]])

    # Process textual features using TF-IDF vectorization
    new_textual_features = tfidf_vectorizer.transform([""])  # Placeholder, you need to input text here

    # Combine structured and textual features
    new_features = np.hstack((new_structured_features, new_textual_features.toarray()))

    # Predict using the trained Random Forest model
    predicted_last_stage = loaded_job_model.predict(new_features)

    st.write("Predicted LastStage:", predicted_last_stage[0])
    if predicted_last_stage == 1:
        st.write('You are not fit for the job')
    else:
        st.write('Congratulations! You are fit for the job')



# Define bag of words function
def bagofwords(msg):
    tokens = nltk.word_tokenize(msg)
    tokens = [stemmer.stem(token.lower()) for token in tokens if token not in string.punctuation and token
              not in stop_words]
    binary_msg = [1 if word in tokens else 0 for word in words]
    return np.array(binary_msg)

# Define prediction function
def prediction(msg):
    if "job details" in msg.lower():
        response = "Please share your complete details.\n" \
                   "So that I can answer if you are fit for the job or not.\n" \
                   "Below you have to enter your details. \n" \
                   "This chatbot uses an algorithm for job detection.\n" \
                   "There are 4 input that I will take from you and detect\n" \
                   "you are fit for job or not.\n" \
                   "Below I have mentioned how put your details.\n" \
                   "For QualLevel:\n" \
                   "For MS/MTech type 5 in Quallevel\n" \
                   "For MCA/ME type 4 in Quallevel\n" \
                   "For BE type 3 in Quallevel\n" \
                   "For BSc/BCS type 2 in QualLevel\n" \
                   "For Other type 1 in QualLevel\n" \
                   "For SourceId:\n" \
                   "For Referal type 6 in SourceId\n" \
                   "For Connsultant1 and Consultant2 type 5 in SourceId\n" \
                   "For LinkedIn type 4 in SourceId\n" \
                   "For Naukri type 3 in SourceId\n" \
                   "For Shine type 2 in SourceId\n" \
                   "For Indeed type 1 in SourceId\n" \
                   "For CurrentExperience and CGPA:\n" \
                   "You can type any integer or float between 0 to 10\n"


    else:
        message = bagofwords(msg)
        result = chatbot.predict(np.array([message]))
        result = np.argmax(result, axis=1)
        class_index = list(result)[0]
        current_tag = classes[class_index]

        for intent in data['intents']:
            if intent['tag'] == current_tag:
                response = random.choice(intent['responses'])
                return response
    return response
def main():
    st.title("Welcome to Intelli Recruit Chatbot")
    st.write("Feel free to start a conversation with the chatbot!")
    # Initialize conversation history as an empty list
    conversation_history = []
    while True:
        text = st.text_input("You:", key=next(key_counter))
        if text:
            conversation_history.append(("You", text))

            # Get the chatbot's response
            response = prediction(text)
            conversation_history.append(("Chatbot", response))

            if "job details" in text.lower():
                st.text(response)
                st.text("Please provide the required details to evaluate your job fit:")
                QualLevel = st.text_input("Enter Your QualLevel")
                CGPA = st.text_input("Enter Your CGPA")
                SourceId = st.text_input("Enter Your SourceId")
                CurrentExperience = st.text_input("Enter Your CurrentExperience")

                if st.button("Check if you're fit for the job"):
                    job_details_prediction(QualLevel, CGPA, SourceId, CurrentExperience)
            else:
                st.text(response)




if __name__ == '__main__':
    main()
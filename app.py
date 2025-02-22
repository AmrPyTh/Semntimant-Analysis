import streamlit as st 
import sklearn
import helper
import pickle
import nltk
import os


nltk.download('punkt')
nltk.download('stopwords')


model_path = os.path.join(os.getcwd(), 'model.pkl')
vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))


title = "<h1 style='text-align: center; color: green; white-space: nowrap;'>Semntimant Analysis App using ML</h1>" 
st.markdown(title, unsafe_allow_html=True)
text = st.text_input('Please Enter Your Review')
state = st.button('Predict')
# token = helper.preprocessing_step(text)
# vectorized_data = vectorizer.transform([token])

# prediction = model.predict(vectorized_data)
if state:
    token = helper.preprocessing_step(text)
    vectorized_data = vectorizer.transform([token])
    prediction = model.predict(vectorized_data)

    if prediction == 1:
        st.markdown("<h3 style='text-align: center; color: green;'>Positive Review</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: red;'>Negative Review</h3>", unsafe_allow_html=True)

import streamlit as st 
import sklearn
import helper
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

model = pickle.load(open(r'F:\Doc\Ai\ODC\NLP\model.pkl', 'rb'))
vectorizer = pickle.load(open(r'F:\Doc\Ai\ODC\NLP\vectorizer.pkl', 'rb'))

title = "<h1 style='text-align: center; color: green; white-space: nowrap;'>Semntimant Analysis App using ML</h1>" 
st.markdown(title, unsafe_allow_html=True)
text = st.text_input('Please Enter Your Review')
state = st.button('Predict')
token = helper.preprocessing_step(text)
vectorized_data = vectorizer.transform([token])

prediction = model.predict(vectorized_data)
if state:
    if prediction == 1:
        text_1 = "<h3 style='text-align: center; color: green; white-space: nowrap;'>Positive Review</h3>" 
        st.markdown(text_1, unsafe_allow_html=True)
    else:
        text_2 = "<h3 style='text-align: center; color: green; white-space: nowrap;'>Negative Review</h3>" 
        st.markdown(text_2, unsafe_allow_html=True)
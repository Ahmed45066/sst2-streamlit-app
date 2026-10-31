import streamlit as st
from transformers import pipeline

st.title("sentiment classifier")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="AhmedRafiq/my-sst2-model")

clf = load_model()

text = st.text_area("write a sentence")

if st.button("check"):
    result = clf(text)[0]
    st.write(result["label"], round(result["score"], 3))

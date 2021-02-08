import streamlit as st
import spacy
from spacy import displacy
from textblob import TextBlob
from spacy import displacy
#import en_core_web_lg
from transformers import pipeline
import spacy.cli
from spacy_streamlit import visualize_parser
import gensim 
#from svlearn.nlp.keywords.keyword_extraction import do_find_keywords, rake_keywords, gensim_keywords
import pandas as pd 
from pandas import DataFrame 
st.title("Explore Natural Language Processing using different models")

in_model_name=st.sidebar.selectbox("Select Model",("Spacy","Transformer"))
if in_model_name=="Spacy":
    in_operation=st.sidebar.selectbox("Select Operation",("Translate","Lemmatize","Recognize NE","Dependency Diagram","Detect Keywords","Detect Language","POS Tagging"))
elif in_model_name=="Transformer":
    in_operation=st.sidebar.selectbox("Select Operation",("Sentiment-Analysis","Translate","Summarize","Recognize NE"))

if in_operation=="Translate":
    in_language=st.sidebar.selectbox("Select Language",("Hindi","Spanish"))

in_text_input=st.text_area(label='Enter text here ...', value="Hello", height=10, max_chars=5000)

def translate(text_input, language):
    blob = TextBlob(text_input)
    if language=="Hindi":
        translation = str(blob.translate(to='hi'))
    elif language=="Spanish":
        translation = str(blob.translate(to='es'))
    else:
        translation=""    
    return translation

def detectLanguage(in_text_input):
    blob = TextBlob(in_text_input)
    lang = blob.detect_language()
    return lang

def init_model(model_name, operation):
    if model_name=="Spacy":
        nlp = spacy.load('en_core_web_lg')
    elif model_name =="Transformer":
        if in_operation=="Sentiment-Analysis":
            nlp = pipeline("sentiment-analysis",device=-1)
        elif in_operation=="Summarize":
            nlp = pipeline("summarization",device=-1)
        elif operation =="Recognize NE":
            nlp = pipeline("ner")
        #elif in_operation=="Translate":
        #    if in_language=="Hindi":
        #        nlp = pipeline("translation_en_to_hi")
        #    elif in_language=="Spanish":
        #        nlp = pipeline("translation_en_to_es")    
        #    else:
        #        pass     
    else:
        pass        
    return nlp  


def process_nlp(model_name,operation, text_input,model):
    if model_name=="Spacy":
        if operation=="Lemmatize":
            doc = model(text_input)
            for token in doc:
                st.write(f"{token.text:<15} => {token.lemma_}  => {token.pos_}")    
        elif operation=="POS Tagging":
            doc = model(text_input)
            for token in doc:
                st.write(f"{token.text:<15} => {token.pos_}")    
        elif operation =="Recognize NE":
            doc = model(text_input)
            for token in doc:
                st.write(f'{token.text:<15} => {token.pos_:>15}=> {token.tag_:>15}')
        elif operation=="Dependency Diagram":
            doc=model(text_input)
            visualize_parser(doc,title="",key=1)
            pass
    elif model_name =="Transformer":
        if in_operation=="Sentiment-Analysis":
            results = model(text_input)
            result_row = results[0]
            result_label = result_row['label']
            result_score = result_row['score']
            st.write(f"Sentiment : = {result_label}")
            st.write(f"Sentiment Analysis Score : = {result_score}")
        elif in_operation=="Summarize":
            result_summary = model(text_input,max_length=300,min_length=100, do_sample=False)
            st.write(result_summary)
        elif operation =="Recognize NE":
            st.write(model(text_input))

        else:
            pass
    else:    
        pass


if in_operation=="Translate":
    translated_text = translate(in_text_input, in_language)
    st.write(translated_text)
elif in_operation=="Detect Keywords":
    # Keyword Detection is currently not working
    st.write(gensim_keywords(in_text_input))
elif in_operation=="Detect Language":
    detected_lang_code=detectLanguage(in_text_input)
    if detected_lang_code=="en":
        detected_language="English"
    elif detected_lang_code=="hi":
        detected_language="Hindi"
    st.write(f"Text Language is : ({detected_lang_code}){detected_language}")
else:
    model = init_model(in_model_name, in_operation)
    process_nlp(in_model_name,in_operation,in_text_input,model)

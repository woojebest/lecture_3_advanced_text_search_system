## 220418, 검색기반 챗봇서비스, wygo
# ref: https://github.com/kairess/mental-health-chatbot

## 설치
# pip install streamlit
# pip install faiss
# pip install sentence_transformers

##실행
# streamlit run chatbot.py


import streamlit as st
import torch
import json
from sentence_transformers import SentenceTransformer
import faiss
import time
import pandas as pd
import numpy as np

st.set_page_config(page_title="검색기반챗봇 실습", page_icon="🤖")

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    
    df_ = pd.read_csv('data_wellness_dataset.csv')

    chatbot_class = df_['구분'].to_list()
    chatbot_input = df_['유저'].to_list()
    chatbot_output = df_['챗봇'].to_list()
    chatbot_embedding = df_['embedding'].to_list()
    assert len(chatbot_class) == len(chatbot_input) == len(chatbot_output) == len(chatbot_embedding)
    encoded_data = np.array(chatbot_embedding)
    
    index = faiss.read_index('chatbot_index')  # load indexer
    return index, chatbot_class, chatbot_input, chatbot_output

model = cached_model()
index, chatbot_class, chatbot_input, chatbot_output = get_dataset()

query = '요즘 머리가 아프고 너무 힘들어'

# print('News 검색')
# query = str(input())
# result_idx = search(query)
# print('results :')
# for idx in result_idx:
#     print('\t=> %s (%s)'%(chatbot_output[idx], chatbot_class[idx]))
    
# Title for the page and nice icon
st.title('검색기반 챗봇')
st.header('KAERI 인공지능 미니석사과정 6주차 실습')
st.markdown("[❤️빵형의 개발도상국](https://www.youtube.com/c/빵형의개발도상국)")

def search(query, K=5):
    t = time.time()
    query_vector = model.encode([query])
    Distance, Index = index.search(query_vector, K)
    final_index = Index.tolist()[0]
    print('totaltime: %.1f sec'%(time.time()-t))

    return final_index


# Form to add your items
with st.form("my_form"):
    #get the models
    query = st.text_area("Source Text", '오늘 하루 너무 피곤하다...',max_chars=200)
    
    
    # Create a button
    submitted = st.form_submit_button("Translate")    

    if submitted:
        result_idx = search(query, K=3)
        for idx in result_idx:
            st.info('\t\t=> %s  (%s)'%(chatbot_output[idx], chatbot_class[idx]))
        
        st.write('검색 기반 챗봇 만들기 쉽죠!??ㅎㅎ')






    
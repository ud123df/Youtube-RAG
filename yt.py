import streamlit as st 
import pandas as pd 
import numpy as np 
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import os 
os.environ ['GROQ_API_KEY'] = "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"

ytt_api = YouTubeTranscriptApi()
# user_input = input("Enter the video ID: ")
user_input = st.text_input("🔞Enter the YT video")
user_input = st.text_input("Enter YouTube URL")
transcript_list=None
if user_input:
    # try:
        user_inputs = user_input.split('=')[1]
        st.write(user_inputs)
        transcript_list = ytt_api.fetch(user_inputs)
        transcript_list = list(transcript_list)

        yt = [i['text'] for i in transcript_list]

    # except Exception:
    #     st.error("Transcript unavailable or blocked for this video.")

##############################
        yt=[]
        for i in transcript_list:
            o=str(i)
            o=o.replace("FetchedTranscriptSnippet","")
            yt.append(o)
            # yt.append(o)
        
        #####################################
        
        sts=""
        for i in range(len(yt)):
            u=yt[0].replace("(", "")
            u=u.replace(")", "").split(",")
            u = yt[i].replace("(", "")
            u=u.replace(")", "")
            g=u.find(", start")
            sts= sts+ u[5:g]
        
        ######################################
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([sts])
        
        ######################################
        
        embeddings = HuggingFaceBgeEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        
        ##################################
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        ####################################
        
        from langchain_groq import ChatGroq
        
        ##################################
        
        llm = ChatGroq(
            model = "meta-llama/llama-4-maverick-17b-128e-instruct", 
            temperature=0.1
        )
        
        
        ###################################
        
        prompt = PromptTemplate(
            template= """
              You are a helpful assistant.
              Answer ONLY from the provided transcript context.
              If the context is insufficient, just say you don't know.
              
              {context}
              Question: {question}
            """,
            input_variables = ['context', 'question']
        )
        
        
        #########################################################
        
        
        question          = st.text_input("Enter the Question 😵 : ")
        retrieved_docs    = retriever.invoke(question)
        
        
        ##########################################
        
        
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # context_text
        
        
        ###########################################
        
        final_prompt = prompt.invoke({"context": context_text, "question": question})
        
        ###########################################
        
        answer = llm.invoke(final_prompt)
        st.success(answer.content)
        
        # header = st.header("hello worlds")
        
        
        
        
        
        
        
        
        
        

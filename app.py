
from utils import *
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import os


def main():
    load_dotenv()

    st.set_page_config(page_title="PDF Parser",
                       page_icon=":books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("PDF Chatbot :books:")
    user_question = st.text_input("Ask a question about your documents:")
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vector_store = get_vector_store(text_chunks)

                st.session_state.conversation = get_conversation_chain(vector_store)



    



    




if __name__ == "__main__":
    main()
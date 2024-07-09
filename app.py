
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
        uploaded_files = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):

                if uploaded_files:
                    all_elements = []
                    for file in uploaded_files:
                        elements = get_pdf_chunks(filename=file)


                text_chunks = get_pdf_chunks(pdf_path)

                vector_store = get_vector_store(text_chunks)

                st.session_state.conversation = get_conversation_chain(vector_store)



    



    




if __name__ == "__main__":
    main()
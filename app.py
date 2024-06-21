
from utils import *
import streamlit as st
from dotenv import load_dotenv



def main():
    load_dotenv()

    st.set_page_config(page_title="PDF Parser",
                       page_icon=":books:")
    
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



    # for multiple files
    # documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])






    




if __name__ == "__main__":
    main()
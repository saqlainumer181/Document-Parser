import nest_asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub



nest_asyncio.apply()


def get_pdf_text(pdf_docs):
    raw_text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text

def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(chunks): 
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain













# def set_llm_model():
#     embed_model = OpenAIEmbedding(model="text-embedding-3-small")
#     llm = OpenAI(model="gpt-3.5-turbo-0125",  num_workers=8)

#     Settings.llm = llm
#     Settings.embed_model = embed_model

#     return llm


# def get_documents_and_objects(documents):

#     llm = set_llm_model()
#     node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
#     nodes = node_parser.get_nodes_from_documents(documents)
#     base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        


# def load_parser():

#     parser = LlamaParse(
#         api_key= os.environ.get("LLAMA_PARSE_API_KEY"),
#         result_type="markdown",  
#         num_workers=4,  
#         verbose=True,
#         language="en",  
#     )

#     return parser
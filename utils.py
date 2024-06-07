import nest_asyncio
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os


nest_asyncio.apply()


def set_openai_model():
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-3.5-turbo-0125",  num_workers=8)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm


def get_documents_and_objects(documents):

    llm = set_openai_model()
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        


def load_parser():

    parser = LlamaParse(
        api_key= os.environ.get("LLAMA_PARSE_API_KEY"),
        result_type="markdown",  
        num_workers=4,  
        verbose=True,
        language="en",  
    )

    return parser
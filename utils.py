import nest_asyncio
from llama_parse import LlamaParse
import os



nest_asyncio.apply()



def load_parser():

    parser = LlamaParse(
        api_key= os.environ.get("LLAMA_PARSE_API_KEY"),
        result_type="markdown",  
        num_workers=4,  
        verbose=True,
        language="en",  
    )

    return parser

from utils import *
from dotenv import load_dotenv



def main():
    load_dotenv()
    parser = load_parser()

    # for single file
    documents = parser.load_data("./test_documents/ML-design-patterns.pdf")

    # for single file
    # documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])






    




if __name__ == "__main__":
    main()
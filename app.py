from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def main():
    load_dotenv()

    # Base interface using streamlit
    st.set_page_config(page_title="ChatGPT project by Real")
    st.header("Organizational ChatGPT 💬")

    # Uploading data
    pdf = st.file_uploader("Upload your data: ", type="pdf")

    #Extract data
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()
        # st.write(text)

        #Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        st.write(chunks)

if __name__ == '__main__':
    main()
    
from dotenv import load_dotenv
import streamlit as st

def main():
    load_dotenv()
    st.set_page_config(page_title="ChatGPT project by Real")
    st.header("Organizational ChatGPT 💬")

    pdf = st.file_uploader("Upload your data: ", type="pdf")

if __name__ == '__main__':
    main()
    
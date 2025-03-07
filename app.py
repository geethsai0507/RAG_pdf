import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Ollama  # Import Ollama LLM




# Load environment variables
load_dotenv()

# Define Ollama-based LLM
def get_llm():
    return Ollama(model="mistral")  # Using Mistral; you can replace it with LLaMA3, etc.

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Get embeddings using Hugging Face's Instructor Model

def get_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # More stable model
    model_kwargs = {'device': 'cpu'}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    return hf_embeddings




# Create a FAISS vector store
def get_vectorstore(text_chunks, embeddings):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Create a conversational retrieval chain
def get_conversation_chain(vectorstore):
    llm = get_llm()  # Use Ollama instead of Hugging Face/OpenAI
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Handle user input
def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs before asking questions.")
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        st.write(f"**{'User' if i % 2 == 0 else 'Bot'}:** {message.content}")


# Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDFs using Local LLM", page_icon="📄")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs Using Local LLM 📄🤖")
    user_question = st.text_input("Ask a question about your PDFs:")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                embeddings = get_embeddings()
                vectorstore = get_vectorstore(text_chunks, embeddings)

                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()

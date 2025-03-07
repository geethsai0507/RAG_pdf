# RAG_pdf

Here’s a **detailed `README.md`** file for your project that clearly explains its functionality, setup, and usage. This will help others (and reviewers for **BUILD for Bharat**) understand your work easily.  

---

# **MultiPDF Chatbot using Local LLMs (Ollama)**
### **Chat with multiple PDFs using a local AI model powered by Ollama & FAISS**

## 🚀 **Overview**
This project is an **AI-powered chatbot** that allows users to upload multiple PDFs and ask **context-aware questions** about their content. Instead of using cloud-based APIs, this chatbot runs **entirely offline** using **Ollama for local LLM inference** and **FAISS for vector-based document retrieval**.  

🔹 **No internet required!**  
🔹 **Supports local LLMs like Mistral, LLaMA3** via **Ollama**  
🔹 **Processes and indexes PDFs for fast, intelligent retrieval**  
🔹 **Uses Streamlit for an interactive chatbot UI**  

---

## 🛠️ **How It Works**
1. **PDF Upload & Processing**: Extracts and splits text from uploaded PDFs.  
2. **Embedding Generation**: Converts document chunks into **vector embeddings** using `sentence-transformers`.  
3. **Vector Storage**: Stores embeddings in **FAISS** for fast similarity search.  
4. **Local AI Model (Ollama)**: Uses an **LLM** (like Mistral or LLaMA3) to generate intelligent responses based on retrieved text.  
5. **User Interaction**: Users ask questions in a **chat-like interface** powered by **Streamlit**.  

---

## 🔧 **Installation & Setup**
### **Step 1: Install Ollama (Local LLM Engine)**
Ollama is used to run LLMs **offline**. Install it using:  
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Then, pull a **local AI model** (e.g., Mistral or LLaMA3):  
```bash
ollama pull mistral
```

### **Step 2: Clone This Repository**
```bash
git clone https://github.com/your-username/MultiPDF-Chatbot.git
cd MultiPDF-Chatbot
```

### **Step 3: Install Required Python Packages**
Ensure you have **Python 3.8+** installed, then install dependencies:  
```bash
pip install -r requirements.txt
```

### **Step 4: Run the Application**
Start the chatbot using **Streamlit**:  
```bash
streamlit run app.py
```
This will open the **chat interface** in your browser. 🎉  

---

## 📌 **Usage**
1. **Upload PDFs** using the sidebar.  
2. Click **"Process"** to extract and store document embeddings.  
3. Type a **question related to the uploaded PDFs** in the chatbox.  
4. The chatbot **retrieves relevant text** and generates an answer using the **local LLM**.  

---

## 🏗️ **Tech Stack**
- **🧠 LLM**: [Ollama](https://ollama.com/) (supports Mistral, LLaMA3, Falcon)  
- **🔍 Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`  
- **📄 PDF Processing**: `PyMuPDF`  
- **🔎 Vector Database**: `FAISS`  
- **💻 UI**: `Streamlit`  
- **🛠 Backend**: `LangChain`  

---

## 📚 **Example Use Cases**
✅ **Academic Research** – Quickly find answers from research papers  
✅ **Legal Documents** – Ask context-based legal questions  
✅ **Business Reports** – Extract insights from company reports  
✅ **Technical Manuals** – Interact with documentation easily  

---

## 🚀 **Future Improvements**
🔹 **Support for multi-modal AI (PDFs + Images + Tables)**  
🔹 **Advanced fine-tuning for local LLMs**  
🔹 **Improve UI with chatbot memory & history**  

import streamlit as st
import pdfplumber
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

# Initialize LLM
llm = Ollama(model="llama3.2")

# Initialize memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Streamlit page setup
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("AI Interviewer")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
resume_text = ""

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Resume Text", resume_text, height=200)

# Display past chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Candidate Responds
user_input = st.chat_input("Enter Your Response:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Single dynamic prompt
    prompt = f"""
    You are an AI interviewer. 
    The candidate has uploaded a resume: {resume_text if resume_text else "No resume provided"}.
    The candidate responded: {user_input}.
    
    Based on this, generate the next interview question. Keep it engaging and technical.
    """
    ai_response = llm(prompt)
    import re
    ai_response = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()

    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    with st.chat_message("assistant"):
        st.markdown(ai_response)

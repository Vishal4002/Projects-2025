from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import PromptTemplate

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm=Ollama(model="llama3.2",base_url="http://127.0.0.1:11434")

def detect_document_type(text):

    text_lower=text.lower()

    if any(keyword in text for keyword in["agreemet", "contract", "obligations","termination"]):
        return "Agreement"

    elif any(keyword in text for keyword in ["experience", "skills", "education"]):
        return "Resume"
    
    else:
        return "General"
    
summary_templates={
"Agreement": """
        Imagine you are a legal expert explaining this agreement to a friend who has no legal background.
        Summarize **only the most critical legal, compliance, and risk-related points** from the following text:
        '''{text}'''
        Keep the explanation **short, simple, and to the point**, like a friendly legal advisor.
        Use **clear, everyday language** and avoid complex legal terms.
        The goal is to make it **easy to understand** while keeping all key details intact.
    """,
    "Resume": """
        You are a professional recruiter reviewing a candidate’s resume. Identify only the **most essential skills, qualifications, and achievements** from the text:
        '''{text}'''
        Rewrite them in structured, **professionally confident sentences**.
        Keep the summary **precise, impressive, and to the point**.
    """,
    "Story": """
        You are a skilled storyteller summarizing a narrative. Extract only the **key plot points and major character developments** from the text:
        '''{text}'''
        Present them in a **clear and engaging way**, without unnecessary details.
    """,
    "General": """
        You are an expert summarizer. Identify and rephrase only the **most important, high-impact points** from the given text:
        '''{text}'''
        Ensure clarity, **remove fluff**, and keep it **sharp, structured, and meaningful**.
    """
}
    
def create_summary_chain(doc_type):
    prompt_template=PromptTemplate(template=summary_templates[doc_type], input_variables=["text"])
    return LLMChain(llm=llm, prompt=prompt_template)

def summarize_large_text(text):
    text=text.replace("\n"," ").strip() 
    doc_type=detect_document_type(text)
    summary_chain=create_summary_chain(doc_type)
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=splitter.split_text(text)
    mapped_summaries=[summary_chain.run({"text": chunk}) for chunk in chunks]
    final_summary="\n".join(mapped_summaries)
    return {"document_type": doc_type, "summary": final_summary.strip()}


class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize-text")
async def summarize_text(request: SummarizeRequest):
    try:
        result=summarize_large_text(request.text)
        return result
    except Exception as e:
        print(f"Error Occured: {str(e)}")
        return {"error": f"An error occured: {str(e)}"}
    

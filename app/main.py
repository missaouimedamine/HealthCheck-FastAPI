from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import analysis, reports
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.prompts import ChatPromptTemplate

    import warnings
    warnings.filterwarnings("ignore")

    # --- Load on Startup ---
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"use_auth_token":HF_TOKEN}
    )

    vectorstore = FAISS.load_local(
        r"app\api\data\faiss_db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(
        base_url="https://api.mistral.ai/v1",
        api_key=Mistral_API_KEY,
        model_name="mistral-medium"
    )

    # Prompt templates
    prompt_template = ChatPromptTemplate.from_template("""
You are a medical assistant. Summarize the following disease information into a clean dictionary format with keys:
['Title', 'Overview', 'Symptoms', 'Causes', 'Risk factors', 'Complications', 'Prevention',
 'When to see a doctor', 'Diagnosis', 'Treatment', 'Lifestyle and home remedies']

Disease Raw Info:
{context}

Only return a JSON-like Python dictionary with medically accurate content under each key.
""")
    
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical diagnosis assistant.

Use the context below to identify which disease best matches the given symptoms.

Return your response in this format as a JSON array of objects WITHOUT any additional text:
[
  {{
    "disease": "Disease Name",
    "probability": 87
  }},
  {{
    "disease": "Other Likely Disease",
    "probability": 13
  }}
]

### Context:
{context}

### Symptoms:
{question}
"""
)

    # RAG Chains
    diagnosis_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    
    
    

# Initialize the RetrievalQA chain
    info_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
    chain_type="stuff",  # or "map_reduce", etc. depending on what you want
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False
)

    # Attach to app state
    app.state.diagnosis_chain = diagnosis_chain
    app.state.info_chain = info_chain
    

    yield

    # On shutdown (if needed):
    # Cleanup resources

app = FastAPI(
    title="Symptom Checker API",
    description="API for symptom checking and analysis",
    version="0.1.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://healthcheck-fastapi.onrender.com/"],  # Adjust this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
#app.include_router(symptoms.router, prefix="/api/symptoms", tags=["symptoms"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["infos"])
app.include_router(reports.router, prefix="/api/reports", tags=["report"])

@app.get("/")
async def root():
    return {"message": "Symptom Checker API is running"}

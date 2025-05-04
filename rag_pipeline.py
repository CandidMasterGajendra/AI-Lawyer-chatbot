from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# step 1: LLM setup using groq
llm_model = ChatGroq(model='deepseek-r1-distill-llama-70b')

# step 2: Retrieve Docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

# it will return docs -> which will contain filepath, metadata, pageNo, documentNo, chunkNo etc. -> not much usefull, only content is needed
def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# step 3: Answer Question
system_prompt = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""
def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(system_prompt)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})


from gc import set_debug
import os
import tempfile
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.globals import set_verbose
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient

load_dotenv(find_dotenv())
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
openai_key=os.environ.get("OPENAI_API_KEY")
assert openai_key is not None
secret_ai_key=SecretStr(openai_key)
tavily_key=os.environ.get("TAVILY_API_KEY")
tavily_client=TavilyClient(api_key=tavily_key)
# set_debug(True)
# set_verbose(True)




st.title("Learn more about what you want.")
st.text("Upload your own file (only PDF's) and ask questions to become a subject matter expert yourself. We'll help you understand it better.")
st.divider()
uploaded_file= st.file_uploader("choose a file for you to learn moe about...")

@tool
def initialise_rag():
    """Use this to initialise uploaded doc's as rag base to query and search"""
    embeddings=OpenAIEmbeddings(model="text-embedding-ada-002",api_key=secret_ai_key)
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path=tmp_file.name
            
            loader=PyPDFLoader(tmp_file_path)
            docs = loader.load()
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )    
        chunks=text_splitter.split_documents(docs)
        vector_store=Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            
        )
        retriever=vector_store.as_retriever(search_kwargs={"k":1})
        return retriever
    return f"no file uploaded: {None}" # Return None if no file is uploaded or processed

@tool
def web_search(query:str):
  """use this to search the web to find answers based on user query and embeddings retrieved from uploaded pdf in chroma DB"""  
  results=tavily_client.search(query=query)
  return results

system_prompt=(
 """
 
 You are an advanced, multi-step research assistant. Your goal is to provide the most accurate, comprehensive, and up-to-date answers by intelligently combining private knowledge and live web data.

You have access to the following tools:
1. `web_search`: Searches the live internet for recent developments, news, and external verification.

CRITICAL OPERATIONAL PIPELINE:
For every user query, you must follow this strict execution loop:

1. INTERNAL FIRST: Always search the internal knowledge base first. Evaluate the retrieved documents deeply.
2. GAP ANALYSIS: If the vector database yields no results, incomplete answers, or outdated information, identify exactly what is missing.
3. EXTERNAL EXPANSION: Use `web_search` to fill in those specific informational gaps.
4. SYNTHESIS & RE-CHECK: If the web search reveals new keywords, terms, or historical context, pivot back and query `vector_db_search` with these refined terms to check if related internal data was missed.
5. ITERATE: Repeat this loop until you have synthesized the absolute best, most comprehensive answer possible.

RESPONSE FORMAT RULES:
- If an answer is found exclusively in the internal database, flag it as [Internal Source].
- If an answer required web research, clearly cite your external findings alongside internal policies.
- If a conflict arises between internal documents and web data, highlight the contradiction clearly for the user.
- Remain objective, concise, and professional. 
 """
)




with st.spinner("Loading Resources, Please wait..."):
    llm=ChatOpenAI(
    model="gpt-4.1",
    api_key=secret_ai_key,
    temperature=0.0,
    )
    prompt=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
     MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    tools=[web_search,initialise_rag]
    agent=create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools,
        )
    result=AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    
if "messages" not in st.session_state:
    st.session_state["messages"]=[]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])
        
def main():
  if uploaded_file is not None:
     
    user_query=st.chat_input("please enter your Question?")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role":"user","content":user_query})
    
    with st.chat_message("assistant"):
        message_placeholder=st.empty()
        with st.spinner("searching user resources to give best answer..."):
            response=result.invoke({"input":user_query, "chat_history":st.session_state.messages}) 
            if response["output"]:
                st.markdown(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]}) 
 
 
if __name__=="__main__":
 main()           
                            
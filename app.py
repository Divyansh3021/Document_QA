import streamlit as st
from langchain.llms import LlamaCpp
import huggingface_hub
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

st.set_page_config(page_title="DOCAI", layout="wide")
st.markdown(f"""
            <style>
            .stApp{{background-image: url("https://images.pexels.com/photos/924824/pexels-photo-924824.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
            background-attachment: fixed;
            background-size: cover;}}
            </style>
            """, unsafe_allow_html=True)

def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

model_id = "TheBloke/LLaMa-7B-GGML"
model_basename = "llama-7b.ggmlv3.q4_0.bin"

model_path = huggingface_hub.hf_hub_download(repo_id=model_id, filename=model_basename)
print("Model Downloaded!!")
llm = LlamaCpp(model_path=model_path)
embeddings = LlamaCppEmbeddings(model_path = model_path)
llm_chain = LLMChain(llm = llm, prompt = prompt)

st.title("Document Conversation")
uploaded_file = st.file_uploader("Upload the text", type='txt')

if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    # st.write(content)
    file_path = "temp/file.txt"
    write_text_file(content, file_path)   
    
    loader = TextLoader(file_path)
    docs = loader.load()    
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    db = Chroma.from_documents(texts, embeddings)    
    st.success("File Loaded Successfully!!")

    question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?", disabled=not uploaded_file,)    
    if question:
        similar_doc = db.similarity_search(question, k=1)
        context = similar_doc[0].page_content
        query_llm = LLMChain(llm=llm, prompt=prompt)
        response = query_llm.run({"context": context, "question": question})        
        st.write(response)
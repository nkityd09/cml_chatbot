import os
import gradio as gr
import shutil
import random
import time
import warnings

warnings.filterwarnings("ignore")
import textwrap
import langchain
from langchain.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
### Multi-document retriever
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
import glob
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import uuid
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings
from langchain.prompts import PromptTemplate

IP_ADDR=os.environ["VectorDB_IP"]
chroma = chromadb.HttpClient(host=IP_ADDR, port=8000)


class CFG:
    model_name = 'falcon-7' # wizardlm, llama-2, bloom, falcon-40

#access_token = os.environ["HF_TOKEN"]
    
def get_model(model = CFG.model_name):
    
    print('\nDownloading model: ', model, '\n\n')
    
    if CFG.model_name == 'falcon-40':
        tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-40b-instruct')
        
        model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-40b-instruct',
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True
                                                    )
        max_len = 1024
        task = "text-generation"
        T = 0
        
    elif CFG.model_name == 'llama-2':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf") #meta-llama/Llama-2-13b-chat-hf
        
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", #meta-llama/Llama-2-13b-chat-hf
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     #token=access_token
                                                    )
        max_len = 4096
        task = "text-generation"
        T = 0.1

    elif CFG.model_name == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
        
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1",
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                    )
        max_len = 1024
        task = "text-generation"
        T = 0
        
    elif CFG.model_name == 'falcon-7':
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
        
        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct",
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     #low_cpu_mem_usage=True,
                                                     trust_remote_code=True
                                                    )
        max_len = 2048
        task = "text-generation"
        T = 0        
        
    else:
        print("Not implemented model (tokenizer and backbone)")
        
    return tokenizer, model, max_len, task, T

tokenizer, model, max_len, task, T = get_model(CFG.model_name)

pipe = pipeline(
    task=task,
    model=model, 
    tokenizer=tokenizer, 
    max_length=max_len,
    temperature=T,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)



#Uploading Files to target location
target = '/home/cdsw/data/'
def upload_file(files):
    """
    """
    file_paths = [file.name for file in files]
    print(file_paths)
    for file in file_paths:
        shutil.copy(file, target)
    return file_paths



embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") #TODO: Find replacement

langchain_chroma = Chroma(
client=chroma,
collection_name="default",
embedding_function=embedding_function)

retriever = langchain_chroma.as_retriever(search_kwargs={"k": 3, "search_type" : "similarity"})

def create_default_collection():
    """
    Create Default Collection in ChromaDB if no collections exists
    """
    chromadb.create_collection("default")
    return "Created Default Collection"



def collection_lists():
    """
    
    """
    collection_list = []
    chroma_collections = chroma.list_collections()
    if chroma_collections == None:
        create_default_collection()
    else:
        for collection in chroma_collections:
            collection_list.append(collection.name)
    return collection_list

collection_list = collection_lists()

def embed_documents(collection):
    """
    
    """
    loader = DirectoryLoader("/home/cdsw/data/",
                         glob="**/*.pdf",
                         loader_cls=PyPDFLoader,
                         use_multithreading=True)

    documents = loader.load()

    for i in range(len(documents)):
        documents[i].page_content = documents[i].page_content.replace('\t', ' ')\
                                                         .replace('\n', ' ')\
                                                         .replace('       ', ' ')\
                                                         .replace('      ', ' ')\
                                                         .replace('     ', ' ')\
                                                         .replace('    ', ' ')\
                                                         .replace('   ', ' ')\
                                                         .replace('  ', ' ')

#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#    texts = text_splitter.split_documents(documents)
    langchain_chroma = Chroma(
    client=chroma,
    collection_name=collection,
    embedding_function=embedding_function)    

    collection = chroma.get_collection(collection) # Needs to be initialized as LangChain cannot add texts

    # Document is chunked per page. Each page will be an entry in the Vector DB
    for doc in documents: 
        collection.add(
            ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
        )

    pattern = "/home/cdsw/data/*.pdf"
    files = glob.glob(pattern)
    
    output = f"Documents have been embedded: {files}"
    print(output)
    # Search files with .pdf extension in directory and delete them


    # deleting the files with txt extension
    for file in files:
        os.remove(file)
    #Extra Line
    
    return output


##### Experimentatal Code ##### 
def set_retriver(collection_name):
    langchain_chroma = Chroma(
    client=chroma,
    collection_name= collection_name,
    embedding_function=embedding_function)
    
    retriever = langchain_chroma.as_retriever(search_kwargs={"k": 2, "search_type" : "similarity"})
    return retriever


# Build prompt
template = """You are a helpful AI assistant and provide the answer for the question based on the given context.
Context:{context}
>>QUESTION<<{question}
>>ANSWER<<"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

  

def chain(query, retriever):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=set_retriver(retriever), 
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                       verbose=True)
    return qa_chain(query)

def add_text(history, text):
    """
    """
    history = history + [(text, None)]
    return history, ""

def bot(history, collection):
    """
    """
    response = llm_ans(history[-1][0], collection)
    history[-1][1] = response
    return history

def wrap_text_preserve_newlines(text, width=110):
    """
    """
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    """
    """
    result = wrap_text_preserve_newlines(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    return result    

 

def llm_ans(query, collection):
    """
    """
    llm_response = chain(query, collection)
    # print(llm_response['result'])
    sources = []
    for source in llm_response["source_documents"]:
        source_file = source.metadata['source']
        source_file = source_file.replace("/home/cdsw/data/", "")
        sources.append(source_file)
    source_files = "\n".join(sources) 
    ans = llm_response['result'] + "\n \n Relevant Sources: \n" + source_files
    return ans

def reset_state():
    """
    """
    return [], [], None


with gr.Blocks() as demo:
    with gr.Tab("FileGPT"):
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=650)
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                collection_dropdown = gr.Dropdown(
                    collection_list, label="Chroma Collections", info="Choose a Collection to Query",
                    value = "default", max_choices=1
                )
                emptyBtn = gr.Button("Clear History")
        user_input.submit(add_text, [chatbot, user_input], [chatbot, user_input]).then(bot, [chatbot, collection_dropdown], chatbot)
        submitBtn.click(add_text, [chatbot, user_input], [chatbot, user_input]).then(bot, [chatbot, collection_dropdown], chatbot)
        history = gr.State([])
        past_key_values = gr.State(None)
        emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)
        

    with gr.Tab("Upload File"):
        with gr.Row():
            with gr.Column(scale=4):
                file_output = gr.File()
                upload_button = gr.UploadButton("Click to Upload a File", file_types=[".pdf",".csv",".doc"], file_count="multiple")
                upload_button.upload(upload_file, upload_button, file_output)
            with gr.Column(scale=1):
                embed_dropdown = gr.Dropdown(
                    collection_list, label="Chroma Collections", info="Choose a Collection to Query", 
                    value = "default", max_choices=1
                )
                embed_button = gr.Button("Embed Document", variant="primary")
                txt_3 = gr.Textbox(value="", label="Output")
                
                
    embed_button.click(embed_documents, embed_dropdown, show_progress=True, outputs=[txt_3])
    


    
demo.queue()


if __name__ == "__main__":
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT'))) 

    print("Gradio app ready")
    
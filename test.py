from fastapi import FastAPI, HTTPException,Request,File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ctransformers import AutoConfig, Config, AutoModelForCausalLM
from pydantic import BaseModel
import json
import uvicorn
import uuid
import pickle
import os,re
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

class PromptInput(BaseModel):
    prompt: str
class MergePDFRequest(BaseModel):
    files: UploadFile
app = FastAPI()

from huggingface_hub import login
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
conf = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1,
                         batch_size=52, max_new_tokens=3000,
                         context_length=4096,gpu_layers=5,stream=True))
# conf2 = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1,
#                          batch_size=52, max_new_tokens=3000,
#                          context_length=4096,gpu_layers=5))

print("joe")
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_type="mistral", config=conf)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[ "*"],  # this is needed for streaming data header to be read by the client
)


# vectorstore storing summaries
vectorstore = Chroma(
    collection_name="summaries", embedding_function=GPT4AllEmbeddings()
)

# The storage layer for the unchunked list storage wrt summary linked with id_key
store = InMemoryStore()
id_key="doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

def pdf_extraction(file_path: str,file_name:str):
    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=file_path + file_name,
        # Using pdf format to find embedded image blocks
        extract_images_in_pdf=True,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=True,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        # Hard max on chunks
        max_characters=1850,
        new_after_n_chars=1750,
        combine_text_under_n_chars=1000,
        extract_image_block_output_dir=file_path+"figures/"
    )
    return raw_pdf_elements

def give_unchunked_list(raw_pdf_elements:list):
    unchunked_list = []
    for ele in raw_pdf_elements:
        sublist = []
        for orig_ele in ele.metadata.orig_elements:
            metadata = orig_ele.metadata.to_dict()
            filtered_dict = {}
            filtered_dict['text'] = orig_ele.to_dict()['text']
            filtered_dict['type'] = orig_ele.to_dict()['type']
            filtered_dict['page_number'] = metadata['page_number']
            if 'image_path' in metadata:
                filtered_dict['image_path']=metadata['image_path']
            sublist.append(filtered_dict)
        unchunked_list.append(sublist)
    return unchunked_list
# def get_model():
#     conf = AutoConfig(Config(temperature=0.8, repetition_penalty=1.1,
#                          batch_size=52, max_new_tokens=3000,
#                          context_length=4096,gpu_layers=37))
#     llm = cAutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_type="mistral", config=conf)
#     return llm

def generate_output(question):
    formatted_input = f"<s> <INST> {question} </INST>"
    output=""
    for chunk in llm(formatted_input, temperature=0.7,repetition_penalty=1.15, max_new_tokens=1000):
        print(chunk)
        output+=chunk
    return output

def prompt_input(input_text):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. \
    The text chunk which you will receive will be in the following format: {{'text': '', 'type': '', 'page_number': }}. If the type argument has value 'Title', THEN MAKE SURE to include it in your summary. There may be more than one title in the text chunk you will receive; make sure that you give EACH text which has 'Title' type in your summary output. If there is any text in the image, include that too. Ignore all the page numbers. Do not use new line characters. \
    Table or text chunk: {element} """
    return prompt_text.format(element=input_text)

def get_text_summaries(chunklist):
    # chunk_text_list = [i.text for i in raw_pdf_elements]
    lambda_chain = lambda input_list: list(map(lambda text: generate_output(prompt_input(text)), input_list))
    text_summaries = lambda_chain(chunklist)
    return text_summaries



def clean_db():
    store.mdelete(list(store.__dict__['store'].keys()))
    vectorstore.delete(vectorstore.get()['ids'])
    print(list(store.__dict__['store'].keys()))
    print(vectorstore.get()['ids'])
    print("db cleared successfully..")



def save_to_pickle(obj, filename):
    with open(filename, 'wb') as filehandler:
        pickle.dump(obj, filehandler)
def load_from_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
def save_stores_data(text_summaries, id_key, unchunked_list):
    try:
        # Add texts
        print(text_summaries)
        doc_ids = [str(uuid.uuid4()) for _ in text_summaries]
        summary_texts = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(text_summaries)
        ]
        directory = "./datafiles"
        os.makedirs(directory, exist_ok=True)
        save_path_summary=os.path.join(directory, "summary_texts.pkl")
        save_path_docstore_data=os.path.join(directory, "docstore_data.pkl")
        save_to_pickle(summary_texts, save_path_summary)
        save_to_pickle(list(zip(doc_ids, unchunked_list)), save_path_docstore_data)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_stores_data():
    directory = "./datafiles"
    save_path_summary=os.path.join(directory, "summary_texts.pkl")
    save_path_docstore_data=os.path.join(directory, "docstore_data.pkl")
    if not os.path.exists(save_path_summary):
        with open(save_path_summary, 'w') as f:
            pass
    if not os.path.exists(save_path_docstore_data):
        with open(save_path_docstore_data, 'w') as f:
            pass
    summary_texts=load_from_pickle(save_path_summary)
    docstore_data=load_from_pickle(save_path_docstore_data)
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(docstore_data)
    print("loaded existing datafiles from previous session..")

# def similarity_search_retrieval(user_q):
#     retrieved_query_vectorstore=retriever.vectorstore.similarity_search(user_q)
#     return retrieved_query_vectorstore



def prompt_input_impg(user_q, retrieved_info1,retrieved_info2, retrieved_info3):
    prompt_text = """
        You are a helpful, respectful and honest assistant. The user is asking about the given question.
        User Question:{user_q}
        Information1:{retrieved_info1}
        Information2:{retrieved_info2}
        Information3:{retrieved_info3}
        The information provided to you is in the form of a list where each element in the list is a dictionary containing the information having text, page_number, and type.
        Task 1: If applicable, locate and provide the path to the closest image related to the discussion from the list.
        Task 2: provide page number from which the text is taken.
        Instructions:
        Give output in the following json format: image_path:<list of image path for closest image or just list none if it is not right after the text you have used>, page_number: <list of the page number(s) fetched>
        You need to provide the user by outputting the image path(s) relevant to it. You will be provided with image paths in Information1, Information2 and Information3.
        You can output more than one image paths if required.
        Make sure the output is in this json format, dont output anything else other than what asked and mentioned in the above json format."""
    return prompt_text.format(user_q=user_q, retrieved_info1=retrieved_info1,retrieved_info2=retrieved_info2,retrieved_info3=retrieved_info3)

def prompt_input_para(user_q, retrieved_info1,retrieved_info2, retrieved_info3):
    prompt_text = """
        You are a helpful, respectful and honest assistant. The user is asking about the given question.
        User Question:{user_q}
        Information1:{retrieved_info1}
        Information2:{retrieved_info2}
        Information3:{retrieved_info3}
        The information provided to you is in the form of a list where each element in the list is a dictionary containing the information having text, page_number, and type.
        Task:Based on the provided information, generate a concise and accurate answer to the user's question.
        Instructions:
        Use the information provided to craft an answer that directly addresses the user's question.
        If there are multiple relevant sections, prioritize the most pertinent ones for the response.
        You dont need to output any image or image path, or any page number. Just output the answer text, without "Answer:".
        Ensure that the answer is clear, coherent, and free from ambiguity. Try to answer interactively.
        """
    return prompt_text.format(user_q=user_q, retrieved_info1=retrieved_info1,retrieved_info2=retrieved_info2,retrieved_info3=retrieved_info3)



def generate_retrieved_prompt(user_q):
    retrieved_query_vectorstore=retriever.vectorstore.similarity_search(user_q)
    doc_id1=retrieved_query_vectorstore[0].__dict__['metadata']['doc_id']
    doc_id2=retrieved_query_vectorstore[1].__dict__['metadata']['doc_id']
    doc_id3=retrieved_query_vectorstore[2].__dict__['metadata']['doc_id']
    print(doc_id1,doc_id2,doc_id3)
    prompt_answer=prompt_input_para(user_q=user_q, retrieved_info1=retriever.docstore.store[doc_id1],retrieved_info2=retriever.docstore.store[doc_id2],retrieved_info3=retriever.docstore.store[doc_id3])
    prompt_impg=prompt_input_impg(user_q=user_q, retrieved_info1=retriever.docstore.store[doc_id1],retrieved_info2=retriever.docstore.store[doc_id2],retrieved_info3=retriever.docstore.store[doc_id3])
    return [prompt_answer,prompt_impg]

def generate_retrieve_impg(prompt):
    # output=llm(prompt, temperature=0.7,repetition_penalty=1.15, max_new_tokens=2048)
    # print(output)
    out=""
    for chunk in llm(prompt, temperature=0.7,repetition_penalty=1.15, max_new_tokens=1000):
            # print(chunk)
            out+=chunk
    return out

def regex_impg(inp):
    # import re
    output = inp
    image_path_pattern = r'figure-(\d+-\d+)\.jpg'
    image_path_match = re.findall(image_path_pattern, output)
    images=[]
    if len(image_path_match):
        for ele in image_path_match:
            image_name="figure-"+ele+".jpg"
            images.append(image_name)
    page_number_pattern = r'"page_numbers?"\s*:\s*\[([0-9\s,"]+)\]'
    page_numbers_match = re.search(page_number_pattern, output)
    page_numbers = []
    if page_numbers_match:
        numbers_group = page_numbers_match.group(1)
        for num in numbers_group.split(','):
            num = num.strip()
            num = num.strip('"')
            if num.isdigit():
                page_numbers.append(int(num))
    page_number_pattern = r'"page_number"\s*:\s*(\d+)'
    page_numbers_match = re.search(page_number_pattern, output)
    if page_numbers_match:
        numbers_group = page_numbers_match.group(1)
        for num in numbers_group.split(','):
            num = num.strip()
            num = num.strip('"')
            if num.isdigit():
                page_numbers.append(int(num))
    print(images,"images")
    print(page_numbers)

    return [images,page_numbers]
# def process_output(user_q):
#     generate_output(user_q)
#     sus=json.loads(generate_output(user_q))

load_stores_data()



# api for answering query
@app.post("/conversation")
async def generate_conversation(req:Request):
    try:
        
        inp=await req.json()
        messages=inp["messages"]
        last_message=messages[len(messages)-1]['content']
        # print(last_message)
        final_prompt=generate_retrieved_prompt(last_message)
        # print(final_prompt)
        # final_prompt=f"<s><INST> Answer/reply in Brief \n Question: {inp['messages'][len(inp['messages'])-1]['content']} </INST>"
        # impg=generate_retrieve_impg(final_prompt[1])
        def generate_text(prompt: str):
            for chunk in llm(prompt, temperature=0.7,repetition_penalty=1.15, max_new_tokens=2048):
                print(chunk)
                yield chunk
        output=generate_text(final_prompt[0])
        return StreamingResponse(output, media_type="text/event-stream")

    except Exception as e:
        print("joe"+e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/impg")
async def generate_impg(req:PromptInput):
    try:
        last_message=req.prompt
        # llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_type="mistral", config=conf2)
        # messages=inp["messages"]
        # print(messages)
        # last_message=messages[len(messages)-1]['content']
        print(last_message)
        final_prompt=generate_retrieved_prompt(last_message)
        print(final_prompt[1])
        # final_prompt=f"<s><INST> Answer/reply in Brief \n Question: {inp['messages'][len(inp['messages'])-1]['content']} </INST>"
        impg=generate_retrieve_impg(final_prompt[1])
        rege=regex_impg(impg)
        page_number=list(set(rege[1]))
        images=list(set(rege[0]))
        print("fetched image and page number: ",rege)
        return {'page_number':page_number,'images':images}

    except Exception as e:
        print("errorimg",e)
        raise HTTPException(status_code=500, detail=str(e))

#  api for uploading file and processing it
@app.post("/upload")
async def upload_file(pdf: UploadFile = File(...)):
    try:
        file_path=os.path.join("client/public/","file_merged.pdf")
        with open(file_path, "wb") as f:
            f.write(await pdf.read())
        print("file saved")
        raw_pdf_elements=pdf_extraction(file_path="client/public/",file_name="file_merged.pdf")
        print("pdf extract success")
        unchunked_list=give_unchunked_list(raw_pdf_elements=raw_pdf_elements)
        print("unchunked list success")
        chunk_text_list = [i.text for i in raw_pdf_elements]
        text_summaries=get_text_summaries(chunklist=unchunked_list) #pass either unchunked_list or chunk_text_list
        print("text summarries success")
        clean_db()
        save_stores_data(text_summaries=text_summaries,id_key=id_key,unchunked_list=unchunked_list)
        load_stores_data()
        print("load db success")
        return {"message": "200"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

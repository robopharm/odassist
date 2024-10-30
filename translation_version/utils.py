# Auther: Kyuwan Choi, Gilead Sciences, October 10, 2024
import os
from dotenv import load_dotenv
import boto3
import wave
import pyaudio
from scipy.io import wavfile
import numpy as np
from langchain.llms.bedrock import Bedrock
from langchain_aws import ChatBedrock
import whisper
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
#from langchain_groq import ChatGroq
from botocore.exceptions import BotoCoreError
from langchain_core.prompts import ChatPromptTemplate

from gtts import gTTS
import pygame
from langchain_openai import AzureChatOpenAI
#import asyncio
import json
from botocore.exceptions import ClientError
from langchain_community.chat_models import BedrockChat
#import fitz  # PyMuPDF for PDF handling
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from util_token import TokenCounterHandler
from pprint import pprint
from typing import List
from langchain_aws import BedrockEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import time
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch

load_dotenv()

#groq_api_key = os.getenv("GROQ_API_KEY")
session = boto3.Session()
bedrock_client = session.client('bedrock-runtime', region_name='us-west-2')

my_region = 'us-west-2'
service = 'es'
index_name = 'index_name'
# us-dna-gpro-agt__gpro_smartsearch'
host_url = 'host_url'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, my_region, service, session_token=credentials.token)

embed_model = "amazon.titan-embed-text-v2:0"
region = "us-west-2"
index_name = 'index_name'
embedding = BedrockEmbeddings(
    region_name=region,
    model_id=embed_model
)
client = OpenSearchVectorSearch(
    opensearch_url=[host_url],
    index_name=index_name,
    embedding_function=embedding,
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection,
    timeout=60,
)


def search_index(index_name, search_query) -> List:
    response = client.search(index=index_name, body=search_query)
    return [hit for hit in response["hits"]["hits"]]


class CustomRetriever(BaseRetriever):
    k: int
    index_name: str

    def _get_relevant_documents(
            self, question: str) -> List[Document]:
        embeddings = embedding
        osearch = client
        ################
        if self.k > 2:
            knn = self.k
        else:
            knn = 2

        docs = osearch.similarity_search_with_score(
            question,
            search_type="script_scoring",
            space_type="cosinesimil",
            vector_field="vector_field",
            text_field="text",
            score_threshold=1
        )

        return docs

llm_model = "gpt-4o"
#llm_model = "gpt-4"
temperature = 0
# max_tokens = 1000 #default was 500
os.environ["AZURE_OPENAI_API_KEY"] = "API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "endpoint"
show_intermediate_steps = True
llm = AzureChatOpenAI(model_name=llm_model, temperature=temperature, openai_api_version="2024-08-01-preview",
                          azure_deployment="gpt-4o", )
num_chunks = 3
c = CustomRetriever(index_name=index_name, k=num_chunks)

import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
start = time.time()
question_router = prompt | llm | JsonOutputParser()

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
start = time.time()
retrieval_grader = prompt | llm | JsonOutputParser()
question = "What is naloxone"
response = c._get_relevant_documents(question="What is naloxone")
total_documents = ""
for i in range(0,num_chunks):
    total_documents += response[i][0].page_content
docs = total_documents

print(retrieval_grader.invoke({"question": question, "document": docs}))

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise in plain English of the 6th-8th grade reading level<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
start = time.time()
rag_chain = prompt | llm | StrOutputParser()
# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

# Prompt
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
start = time.time()
hallucination_grader = prompt | llm | JsonOutputParser()
hallucination_grader_response = hallucination_grader.invoke({"documents": docs, "generation": generation})

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
start = time.time()
answer_grader = prompt | llm | JsonOutputParser()
answer_grader_response = answer_grader.invoke({"question": question,"generation": generation})
print(answer_grader_response)

from langchain_community.tools import DuckDuckGoSearchRun

web_search_tool = DuckDuckGoSearchRun()

from typing_extensions import TypedDict
from typing import List

### State

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]


from langchain.schema import Document

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    # documents = vector_search_as_retriever.invoke(question)
    # return {"documents": documents, "question": question}

    # docs = vectorstore_faiss.similarity_search(question, k=3)
    # documents = docs[0].page_content
    response = c._get_relevant_documents(question=question)
    total_documents = ""
    for i in range(0, num_chunks):
        total_documents += response[i][0].page_content
    documents = total_documents
    return {"documents": documents, "question": question}


#
def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


#
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


#
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    #documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    print(docs)
    # web_results = "\n".join([d["content"] for d in docs])
    web_results = docs
    web_results = Document(page_content=web_results)
    # if documents is not None:
    #     documents.append(web_results)
    # else:
    documents = [web_results]
    return {"documents": documents, "question": question}


#
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

from langgraph.graph import END, StateGraph, START
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)
import time
app = workflow.compile()

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=4):
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'C:/temp_audio_chunk.wav'
    #temp_file_path = 'temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write audio frames

def load_whisper():
    model = whisper.load_model("base")
    return model


def transcribe_audio(model, file_path):
    print("Transcribing...")
    # Print all files in the current directory
    print("Current directory files:", os.listdir())
    if os.path.isfile(file_path):
        results = model.transcribe(file_path) # , fp16=False
        return results['text']
    else:
        return None

def load_prompt():

    input_prompt = """

    Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide concise and short answers not more than 20 words.

    {context}

    Question: {question}

    Assistant:"""

    return input_prompt

def invoke_llama2(bedrock_runtime_client, prompt):
    prompt = load_prompt()
    try:
        body = {
            "prompt": prompt,
            "temperature": 0,
            "top_p": 0.9,
            "max_gen_len": 512,
        }

        ## Change Llama 3.1 model id from bedrock
        model_id = 'meta.llama3-1-405b-instruct-v1:0'
        response = bedrock_runtime_client.invoke_model(
            modelId=model_id, body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        completion = response_body["generation"]

        return completion

    except ClientError:
        print("Couldn't invoke Llama 3")
        raise


def load_llm():

    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        client=bedrock_client,
        model_kwargs=dict(temperature=0),
        # other params...
    )

    return llm


def get_response_llm(user_question, memory):
    model = AzureChatOpenAI(model_name=llm_model, temperature=temperature, openai_api_version="2024-08-01-preview",
                            azure_deployment="gpt-4o", )

    template = "Please translate the sentence into English {Korean}. Provide only the translated English in your response."
    prompt_template = ChatPromptTemplate.from_template(template)

    prompt = prompt_template.invoke({"Korean": user_question})
    result = model.invoke(prompt)
    #print(result.content)
    user_question = result.content

    inputs = {"question": user_question}

    print(user_question)

    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")

    model = AzureChatOpenAI(model_name=llm_model, temperature=temperature, openai_api_version="2024-08-01-preview",
                            azure_deployment="gpt-4o", )

    template = "Please translate the sentence into Spanish {English}. Provide only the translated Spanish in your response."
    prompt_template = ChatPromptTemplate.from_template(template)

    prompt = prompt_template.invoke({"English": value["generation"]})
    result = model.invoke(prompt)
    # print(result.content)
    final_result = result.content

    return final_result

def play_text_to_speech(text, language='es-us', slow=False):
    # Generate text-to-speech audio from the provided text
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the generated audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    # Initialize the pygame mixer for audio playback
    pygame.mixer.init()

    # Load the temporary audio file into the mixer
    pygame.mixer.music.load(temp_audio_file)

    # Start playing the audio
    pygame.mixer.music.play()

    # Wait until the audio playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Control the playback speed

    # Stop the audio playback
    pygame.mixer.music.stop()

    # Clean up: Quit the pygame mixer and remove the temporary audio file
    pygame.mixer.quit()
    os.remove(temp_audio_file)
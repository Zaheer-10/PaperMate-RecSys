from io import BytesIO
import pickle
from pathlib import Path
import re
from urllib.parse import urlparse

# Text-Summarization
import PyPDF2
import requests
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime, time
from transformers import pipeline
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from django.shortcuts import render
from .models import Paper , RecentPaper 
from django.shortcuts import render, get_object_or_404
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter

#display
import base64

#QA
import os
import time
import shutil
from urllib.parse import urlparse
from django.core.cache import cache
from django.http import JsonResponse
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from .pdf_constants  import CHROMA_SETTINGS
from langchain.llms import GPT4All, LlamaCpp
from django.shortcuts import render, get_object_or_404
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from .ingest import process_documents, does_vectorstore_exist
from transformers import AutoModelForCausalLM


load_dotenv()

# # Text-Summarization Models
# checkpoint = r"C:\Users\soulo\MACHINE_LEARNING\PaperMate\PaperMate_ui\GUI\LaMini-Flan-T5-248M"
# # checkpoint = Path.cwd() / "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint , local_files_only=True)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint , local_files_only=True)
# pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=512, min_length=50)

# checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
# pipe_sum = pipeline(
#     'summarization',
#     model=base_model,
#     tokenizer=tokenizer,
#     max_length=512,
#     min_length=50
# )
pipe_sum = pipeline("summarization", model="MBZUAI/LaMini-Flan-T5-248M" ,max_length = 512 ,min_length=50 )


# QA Requirements

# model_path = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.3-groovy")

#env
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50

#models
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

# -------------------------------------------------------Home page------------------------------------------------------------------------------------

def index(request):
    
    """
    Render the index page of the Django web application.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content of the index page with recent research papers categorized for display.
    steps:
         1. Retrieving recent papers from different categories and passing through the context
    """
    
    ml_papers = RecentPaper.objects.filter(category='cs.CL').order_by("-published_date")[:3]
    nlp_papers = RecentPaper.objects.filter(category='cs.LG').order_by("-published_date")[:3]
    ai_papers = RecentPaper.objects.filter(category='cs.AI').order_by("-published_date")[:3]
    cv_papers = RecentPaper.objects.filter(category='cs.CV').order_by("-published_date")[:3]

    # print("ML Papers Count:", ml_papers.count())
    # print("NLP Papers Count:", nlp_papers.count())
    # print("AI Papers Count:", ai_papers.count())
    # print("CV Papers Count:", cv_papers.count())
    
    context = {
        'ml_papers': ml_papers,
        'nlp_papers': nlp_papers,
        'ai_papers': ai_papers,
        'cv_papers': cv_papers,
    }
    return render(request, 'index.html', context)


# ----------------------------------------------search_papers-------------------------------------------------------------------------------------------
def search_papers(request):
    """
    Search for relevant research papers using a user query and recognized speech, and display recommended papers.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content showing recommended research papers based on the user's query.
    """
    try:
        # Setting up paths for pre-trained models and data.
        PATH_SENTENCES = Path.cwd() / "Models/Sentences"
        PATH_EMBEDDINGS = Path.cwd() / "Models/Embeddings"


        if request.method == 'POST':
            # Retrieving user input and recognized speech.
            query = request.POST.get('query', '').strip()
            recognized_text = request.POST.get('recognized_text', '').strip()

            # Combining query and recognized speech if available.
            if recognized_text:
                query += ' ' + recognized_text
        
            # Check if either the query or recognized_text is empty
            if not query:
                return render(request, 'index.html', {'error_message': 'Please enter a query or use speech input.'})
            
            # Load pre-trained SentenceTransformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings_path = PATH_EMBEDDINGS / "Embeddings.pkl"
            sentences_path = PATH_SENTENCES / "Sentences.pkl"
            
            # Load pre-calculated sentence embeddings
            with open(sentences_path, 'rb') as f:
                sentences_data = pickle.load(f)
            with open(embeddings_path, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            # Generate a prompt template based on the user query
            prompt_template = f"Could you kindly generate top ArXiv paper recommendations based on: '{query}'? Your focus on recent research and relevant papers is greatly appreciated."
            
            # Encoding user query and calculating cosine similarity.
            query_embedding = model.encode([prompt_template])
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings_data)[0]
            
            # Get indices of top 4 similar papers
            top_indices = cosine_scores.argsort(descending=True)[:4]
            top_indices = top_indices.cpu().numpy()  # Convert to numpy array
            top_paper_titles = [sentences_data[i.item()] for i in top_indices]  # Access elements using integer indices
            
            # Get paper details from the database
            recommended_papers = Paper.objects.filter(title__in=top_paper_titles)
            
            search_error = len(recommended_papers) == 0
            
            return render(request, 'recommendations.html', {'papers': recommended_papers, 'recommended_papers': recommended_papers, 'search_error': search_error})
            
    except Exception as e:
        # print( f"An error occurred: {str(e)}")
        # Handle exceptions gracefully and provide an error message
        return render(request, 'index.html', {'error_message': f"An error occurred: {str(e)}"})
    
    return render(request, 'index.html')



# ----------------------------------------------Recommendations-----------------------------------------------------------------------------------------

def recommendations(request):
    """
    Retrieve and render all research paper recommendations for display on the recommendations page.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content displaying all recommended research papers.
    """
    
    # Retrieving all papers from the 'Paper' model.
    papers = Paper.objects.all()
    return render(request, 'recommendations.html', {'papers': papers})

# -------------------------------------Summarization----------------------------------------------------------------------------------------------------

def dynamic_chunk_text(text, max_chunk_size):
    chunks = []
    chunk_start = 0
    while chunk_start < len(text):
        chunk_end = min(chunk_start + max_chunk_size, len(text))
        chunks.append(text[chunk_start:chunk_end])
        chunk_start = chunk_end
    return chunks

def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def validate_url(url):
    parsed_url = urlparse(url)
    return parsed_url.scheme and parsed_url.netloc and parsed_url.path.endswith('.pdf')

def summarize_paper(request, paper_id):
    """
    Retrieve and render recommended research paper summaries for display on the summarization page r.

    Args:
        request (HttpRequest): The HTTP request made by the user(summarize).
        paper_id (int): Identifier for the specific paper being summarized (corresponding Paper ID).

    Returns:
        HttpResponse: The rendered HTML content displaying the summarized research paper.
    """
    try:
        paper = get_object_or_404(Paper, ids=paper_id)
        if paper.summary :
            context = {
                'paper': paper,
                'result': paper.summary
            }
            return render(request, 'summarization.html', context)
        print("Paper not yet in the DB , it may take some time :)")
        pdf_url = f"https://arxiv.org/pdf/{paper.ids}.pdf"
        if not validate_url(pdf_url):
            raise ValueError("Invalid PDF URL")
        response = requests.get(pdf_url)
        pdf_content = response.content
        paper.pdf_content = pdf_content
        paper.save()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        start_page = 1
        end_page = min(start_page + 5, len(pdf_reader.pages))
        print("Extracting and splitting the text")
        extracted_text = "".join(page.extract_text() for page in pdf_reader.pages[start_page:end_page])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_text(extracted_text)
        final_text = "".join(text_chunks)
        final_text = re.sub(r'[^a-zA-Z0-9\s]', '', final_text)
        final_text = re.sub(r'\S*@\S*\s?', '', final_text)
        final_text = final_text.rstrip()
        print("Cleaned || SUMMARIZING UNDER PROGRESS !")
        result = pipe_sum(final_text)[0]['summary_text']
        
        # result = pipe_sum(final_text)[0]['summary_text']
        # summaries = []
        # for chunk in text_chunks:
        #     max_length = int(len(chunk) * 0.75)
        #     summary = pipe_sum(chunk , max_length=max_length)
        #     summaries.append(summary[0]['summary_text'])
        # result = "\n".join(summaries)
        
        paper.summary = result
        paper.save()
        context = {
            'paper': paper,
            'result': result
        }
        return render(request, 'summarization.html', context)
    
    except Exception as e:
        error_message = f"Oops, something went wrong: {str(e)}"
        suggestions = [
            "While processing lost connection please check your internet connection",
            "Oops! memory overloaded",
            "Extremely sorry! Please try again"
        ]
        context = {
            'search_error': True,
            'error_message': error_message,
            'suggestions': suggestions,
        }
        return render(request, 'recommendation.html', context)
    
    
# -----------------------------------------Display Paper---------------------------------------------------------------------------------------------------
def display(request, paper_id):
    """
    Retrieve and render recommended research paper summaries for display on the summarization page.

    Args:
        request (HttpRequest): The HTTP request made by the user (summarize).
        paper_id (int): Identifier for the specific paper being summarized (corresponding Paper ID).

    Returns:
        HttpResponse: The rendered HTML content displaying the summarized research paper.
    """
    paper = get_object_or_404(Paper, ids=paper_id)
    
    if paper.pdf_content:
        pdf_base64 = base64.b64encode(paper.pdf_content).decode("utf-8")
    else:
        pdf_url = f"https://arxiv.org/pdf/{paper.ids}.pdf"
        try:
            response = requests.get(pdf_url , stream=True)
            response.raise_for_status() 
            pdf_content = response.content
            
            paper.pdf_content = pdf_content
            paper.save()
            
            pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")
        except requests.RequestException as e:
            error_message = f"Oops, something went wrong: {str(e)}"
            context = {
                'search_error': True,
                'error_message': error_message,
            }
            return render(request, 'recommendation.html', context)
        
    context = {
        'pdf_base64': pdf_base64,
        'paper': paper,
    }
    
    return render(request, 'display.html', context)

# ----------------------------------------------------------------------Q&A--------------------------------------------------------------------------------

def get_pdf_filename(paper_id):
    """
    Return the filename of the PDF file corresponding to a given paper ID.

    Args: paper_id (int): 
        The identifier of the paper whose PDF file is requested.

    Returns: str:
        The filename of the PDF file in the format “{paper_id}.pdf”
    """
    return f"{paper_id}.pdf"

def delete_all_documents(folder_path):
    """
    Delete all the PDF files in a given folder(source_documents).

    Args:
        folder_path (str): The path of the folder containing the PDF files to be deleted.

    Returns:
      None: The function does not return anything, but prints a message for each file deleted.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Deleted the PDF -> {file_path}")

def delete_db(paper_id):
    """
    Delete the database folder associated with a given paper ID.

    Args:
        paper_id (int): The identifier of the paper whose database folder is to be deleted.

    Returns:
        None: The function does not return anything, but prints a message before and after deleting the folder.
    """
    db_folder_path =  Path.cwd() / "db"
    if os.path.exists(db_folder_path):
        shutil.rmtree(db_folder_path)
        print("Deleted the 'db' ")    

def is_pdf_file_present(directory, pdf_filename):
    """
    Check if a PDF file with a given filename exists in a given directory.

    Args:
    directory (str): The path of the directory to look for the PDF file.
    pdf_filename (str): The name of the PDF file to check.

    Returns:
    bool: True if the PDF file exists in the directory, False otherwise.
    """
    full_path = os.path.join(directory, pdf_filename)
    return os.path.exists(full_path)

def download_paper(request, paper_id):
    """
    Download the PDF file of a given paper from arXiv and render it on the talk2me page.

    Args:
        request (HttpRequest): The HTTP request made by the user (download).
        paper_id (int): The identifier of the paper whose PDF file is to be downloaded.

    Returns:
        HttpResponse: The rendered HTML content displaying the PDF file of the paper.
    """
    pdf_filename = get_pdf_filename(paper_id)
    file_path = Path.cwd() / "GUI" / "source_documents"
    if is_pdf_file_present(file_path, pdf_filename):
        print("PDF file already exists. Skipping download.")
    else:
        delete_all_documents(file_path)
        delete_db(pdf_filename)
        
    paper = get_object_or_404(Paper, ids=paper_id)
    print("Downloading X...")
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    CUSTOM_SOURCE_DOCUMENTS_PATH = Path.cwd() / "GUI" / "source_documents"
    file_path = os.path.join(CUSTOM_SOURCE_DOCUMENTS_PATH, pdf_filename)

    if os.path.exists(file_path):        
        paper_title = paper.title if paper else "Welcome To The PaperMate Chat"
        print("paper.title" , paper_title)
        
        context = {
            'paper_title': paper_title
        }
        return render(request, 'chat.html', context)

    elif  paper.pdf_content:
        pdf_base64 = base64.b64encode(paper.pdf_content).decode("utf-8")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  
        with open(file_path, 'wb') as f:
            f.write(paper.pdf_content)  
        ingest()

    else:
        response = requests.get(pdf_url , stream=True)
        pdf_content = response.content
        pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)  
        with open(file_path, 'wb') as f:
            f.write(pdf_content)
        ingest()    
    context = {
    'pdf_base64': pdf_base64 ,
    "paper" : paper }

    return render(request, 'talk2me.html', context)


def ingest():  
    """
    Ingest a new document into the local vector store using HuggingFace embeddings and Chroma.

    Args:
        None: The function does not take any arguments.

    Returns:
        None: The function does not return anything, but prints a message when the ingestion is complete.
    """
    
    # Create an instance of HuggingFaceEmbeddings with the specified model name
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vector store at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print("Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print("Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    print("Data Ingestion Complete!")

qa = None;
def readFileContent():
    """
    Read the environment variables and load the model and the vector store for the retrieval-based question answering system.

    Args:
        None: The function does not take any arguments.

    Returns:
        None: The function does not return anything, but modifies the global variable qa.
    """

    global qa
    load_dotenv()

    # Load environment variables
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    model_type = os.environ.get('MODEL_TYPE')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

    # Construct the absolute path to the model based on the Django project's root directory
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    model_relative_path = "GUI/models/ggml-gpt4all-j-v1.3-groovy.bin"
    model_path = os.path.join(base_path, model_relative_path)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    # Create a retriever object from the vector store with the specified number of source documents to retrieve
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = []  # No streaming stdout callback for now
    if model_type == "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    else:
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    # Create a RetrievalQA object from the language model, chain type, retriever, and return source documents option and assign it to qa
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    

def chatbot_response_gpt(request):
    """
    Get the user's message from the GET request and return a JSON response with the answer from the retrieval-based question answering system.

    Args:
        request (HttpRequest): The HTTP request made by the user (chat).

    Returns:
        JsonResponse: A JSON object containing the user's message, the answer, the response time, and the source documents.
    """
    global qa
    # Extract user's message from the GET request
    user_message = request.GET.get('message', '')
    if qa is None:
        readFileContent()

    start = time.time()
    # Get the answer from the chain
    res = qa(user_message)
    answer, docs = res['result'], res['source_documents']
    end = time.time()
    print(f"Time Taken : {end}")
    print("Response :\n", answer)
    context = {
        'user_message': user_message,
        'response': answer,
        'response_time': round(end - start, 2),
        'source_documents': [doc.page_content for doc in docs],
    }
    return JsonResponse(context)


def chatbot_response(request):
    """
    Render the chat page with the title of the paper corresponding to the PDF file in the source directory.

    Args:
        request (HttpRequest): The HTTP request made by the user (chat).

    Returns:
        HttpResponse: The rendered HTML content displaying the chat page with the paper title.
    """
    source_directory = Path.cwd() / "GUI" / "source_documents"
    pdf_files = list(source_directory.glob("*.pdf"))
    
    if pdf_files:
        file_path = pdf_files[0]
        paper_id = extract_id_from_path(file_path)
        
        if paper_id:
            paper = get_object_or_404(Paper, ids=paper_id)
            paper_title = paper.title
        else:
            paper_title = "Welcome To The PaperMate Chat"
    else:
        paper_title = "No PDF file found"
    
    context = {
        'paper_title': paper_title
    }
    
    return render(request, 'chat.html', context)

def extract_id_from_path(file_path):
    """
    Extract the paper ID and version from the file path of a PDF file.

    Args:
        file_path (str): The path of the PDF file as a string.

    Returns:
        str or None: The paper ID and version in the format "{id}v{version}" as a string, or None if the file path does not match the expected pattern.
    """
    # Convert the file_path to a Path object
    path_object = Path(file_path)
    
    # Get the filename without extension
    file_name = path_object.stem
    
    match = re.search(r'(\d+\.\d+v\d+)$', file_name)
    
    if match:
        id_with_version = match.group(1)
        return id_with_version
    else:
        return None
    
# ----------------------------------------------About---------------------------------------------------------------------------------------

def about(request):  
    """
    Render the 'About' page of the Django web application.
    
    Args:
        request (HttpRequest): The HTTP request made by the user.
    
    Returns:
        HttpResponse: The rendered HTML content of the 'About' page.
    """
    return render(request, 'about.html')


# -----------------------------------------------architecture---------------------------------------------------
def architecture(request):
    return render(request , 'architecture.html')

# -------------------------------------------------END---------------------------------------------------------------------------------------------------------------
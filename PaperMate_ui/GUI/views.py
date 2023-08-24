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
from datetime import datetime
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

load_dotenv()

# Text-Summarization Models
# checkpoint = r"C:\Users\soulo\MACHINE_LEARNING\PaperMate\PaperMate_ui\GUI\LaMini-Flan-T5-248M"
checkpoint = Path.cwd() / "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint , local_files_only=True)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint , local_files_only=True)
pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=512, min_length=50)





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


# ----------------------------------------------search_papers---------------------------------------------------------------------------------------
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
        print( f"An error occurred: {str(e)}")
        # Handle exceptions gracefully and provide an error message
        return render(request, 'index.html', {'error_message': f"An error occurred: {str(e)}"})
    
    return render(request, 'index.html')



# ----------------------------------------------Recommendations---------------------------------------------------------------------------------------

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

# -------------------------------------Summarization-------------------------------------------------------

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
        extracted_text = "".join(page.extract_text() for page in pdf_reader.pages[start_page:end_page])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        text_chunks = text_splitter.split_text(extracted_text)
        final_text = "".join(text_chunks)
        final_text = re.sub(r'[^a-zA-Z0-9\s]', '', final_text)
        final_text = re.sub(r'\S*@\S*\s?', '', final_text)
        final_text = final_text.rstrip()
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
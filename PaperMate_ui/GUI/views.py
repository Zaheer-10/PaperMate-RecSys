import pickle
from pathlib import Path
from .models import Paper
from .models import RecentPaper 
from django.shortcuts import render
from sentence_transformers import SentenceTransformer, util


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

from . import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

# Configure URL patterns for the web application
urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('search/', views.search_papers, name='search_papers'), 
    path('recommendations/', views.recommendations, name='recommendations'), 
    path('summarize/<str:paper_id>/', views.summarize_paper, name='summarize_paper'),
    path('display/<str:paper_id>/', views.display, name='display'),
    path('download_paper/<str:paper_id>/', views.download_paper, name='download_paper'),
    path('chatbot_response/', views.chatbot_response, name='chatbot_response'),
    path('chatbot_response_api/', views.chatbot_response_gpt, name='chatbot_response_gpt'),
    path('about/', views.about, name='about'), 
    path('architecture/', views.architecture, name='architecture'), 

]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

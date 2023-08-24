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
]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

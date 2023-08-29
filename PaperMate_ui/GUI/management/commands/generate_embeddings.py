import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from django.core.management.base import BaseCommand
from ...models import Paper  

# Paths to folders
PATH_EMBEDDINGS = r"C:\Users\soulo\PaperMate\PaperMate_ui\Models\Embeddings"
PATH_SENTENCES = r"C:\Users\soulo\PaperMate\PaperMate_ui\Models\Sentences"

class Command(BaseCommand):
    help = 'Generate embeddings and sentence embeddings for paper titles'

    def handle(self, *args, **kwargs):
        # Load SentenceTransformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Get titles from the database
        # all_paper_titles = Paper.objects.values_list('title', flat=True)
        all_paper_titles = list(Paper.objects.values_list('title', flat=True))

        # Create a DataFrame for verification
        # titles_df = pd.DataFrame({'Title': all_paper_titles})
        # self.stdout.write(self.style.SUCCESS("Titles:"))
        # self.stdout.write(titles_df.to_string(index=False))

        # Generate sentence embeddings
        sentence_embeddings = model.encode(all_paper_titles)

        # Save embeddings to the Embeddings.pkl file
        with open(os.path.join(PATH_EMBEDDINGS, 'Embeddings.pkl'), 'wb') as f:
            pickle.dump(sentence_embeddings, f)

        # Save titles to the Sentences.pkl file
        with open(os.path.join(PATH_SENTENCES, 'Sentences.pkl'), 'wb') as f:
            pickle.dump(all_paper_titles, f)

        self.stdout.write(self.style.SUCCESS("Embeddings and sentence embeddings generated and saved."))

# management/commands/fetch_arxiv_data.py
import arxiv
from django.core.management.base import BaseCommand
from ...models import Paper  # Import your Paper model
import pandas as pd
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer

def query_with_keywords(query , client) -> tuple:
    
    """
    Query the arXiv API for research papers based on a specific query and filter results by selected categories.
    
    Args:
        query (str): The search query to be used for fetching research papers from arXiv.
    
    Returns:
        tuple: A tuple containing three lists - terms, titles, and abstracts of the filtered research papers.
        
            terms (list): A list of lists, where each inner list contains the categories associated with a research paper.
            titles (list): A list of titles of the research papers.
            abstracts (list): A list of abstracts (summaries) of the research papers.
            urls (list): A list of URLs for the papers' detail page on the arXiv website.
    """
    
    # Create a search object with the query and sorting parameters.
    search = arxiv.Search(
        query=query,
        max_results=3000,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    
    # Initialize empty lists for terms, titles, abstracts, urls and ids.
    terms = []
    titles = []
    abstracts = []
    urls = []
    ids = []
    
    print("For each result in the search...")
    for res in tqdm(client.results(search), desc=query):
        # Check if the primary category of the result is in the specified list.
        print("Check if the primary category of the result is in the specified list.")
        if res.primary_category in ["cs.CV", "stat.ML", "cs.LG", "cs.AI" ,"cs.CL"]:
            # If it is, append the result's categories, title, summary, url and ids to their respective lists.
            terms.append(res.categories)
            titles.append(res.title)
            abstracts.append(res.summary)
            urls.append(res.entry_id)
            ids.append(res.entry_id.split('/')[-1])

    # Return the four lists.
    return terms, titles, abstracts, urls , ids


class Command(BaseCommand):
    help = 'Fetches arXiv data and syncs with PostgreSQL database'
      
    def handle(self, *args, **options):
        # List of query keywords for research topics
        print("List of query keywords for research topics")
        query_keywords = [
            # ... (a long list of keywords)
             "\"large language models\"",
        ]
        
        # Create an arXiv API client
        print("Create an arXiv API client")
        client = arxiv.Client(num_retries=20, page_size=50)
        
        # Initialize lists to store data from queries
        all_titles = []
        all_abstracts = []
        all_terms = []
        all_urls = []
        all_ids = []

        print("Query the API for each keyword and accumulate results")
        # Query the API for each keyword and accumulate results
        for query in query_keywords:
            terms, titles, abstracts, urls, ids = query_with_keywords(query, client)
            all_titles.extend(titles)
            all_abstracts.extend(abstracts)
            all_terms.extend(terms)
            all_urls.extend(urls)
            all_ids.extend(ids)
        
        # Loop through fetched data and insert into the database
        print("Loop through fetched data and insert into the database")
        for term, title, abstract, url, paper_id in zip(all_terms, all_titles, all_abstracts, all_urls, all_ids):
            print("Check if the paper already exists in the database based on its paper_id")
            if not Paper.objects.filter(ids=paper_id).exists():
                print("If not, create a new Paper instance and save it to the database")
                paper = Paper(
                    title=title,
                    abstract=abstract,
                    terms=term,
                    url=url,
                    ids=paper_id,
                    summary=None,  # Setting summary to initial null value
                    pdf_content=None  # Setting pdf_content to initial null value
                )
                paper.save()
                self.stdout.write(self.style.SUCCESS(f'Successfully added paper: {title}'))
            else:
                self.stdout.write(self.style.NOTICE(f'Paper already exists: {title}'))

        self.stdout.write(self.style.SUCCESS('Data fetch and sync completed'))

import requests
from datetime import datetime
from django.utils import timezone
import xml.etree.ElementTree as ET
from GUI.models import RecentPaper
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    """
    Custom management command to fetch and update recent papers.
    
    Usage:
    python manage.py update_recent_papers
    """
    
    help = 'Fetch and update recent papers'
    
    
    # Define the command's functionality.
    def handle(self, *args, **options):
        """
        Execute the custom command to fetch and update recent papers.
        
        Args:
            *args: Additional arguments.
            **options: Command options.
        """
        url = "http://export.arxiv.org/api/query?search_query=cat:cs.statML+OR+cat:cs.AI+OR+cat:cs.CL+OR+cat:cs.CV&sortBy=lastUpdatedDate&sortOrder=descending&max_results=150"
        response = requests.get(url)

        if response.status_code == 200:
            print("Got 200 response")
            root = ET.fromstring(response.content)

            RecentPaper.objects.all().delete()  # Clear existing data
            print("Deleted recent once")
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                category = entry.find("{http://www.w3.org/2005/Atom}category").attrib["term"]
                if category in ['cs.LG', 'cs.CL','cs.AI', 'cs.CV' ]:
                    paper = {
                        "title": entry.find("{http://www.w3.org/2005/Atom}title").text,
                        "category": category,
                        "link": entry.find("{http://www.w3.org/2005/Atom}id").text,
                        "authors": ", ".join([author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]),
                        "published_date": datetime.strptime(entry.find("{http://www.w3.org/2005/Atom}published").text, "%Y-%m-%dT%H:%M:%SZ"),
                    }
                    print("Creating paper:", paper)  # Add this line to see which papers are being created
                    RecentPaper.objects.create(**paper)

            self.stdout.write(self.style.SUCCESS('Recent papers updated successfully'))
        else:
            self.stderr.write(self.style.ERROR('Failed to fetch papers'))

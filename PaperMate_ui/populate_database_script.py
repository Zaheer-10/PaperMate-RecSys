import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PaperMate_ui.settings')
django.setup()

from GUI.models import Paper

def main():
    Paper.populate_database()

if __name__ == '__main__':
    main()

import re

import re

# Given path
file_path = r"C:\Users\soulo\PaperMate\PaperMate_ui\GUI\source_documents\2307.06435v1.pdf"

# Extract the ID using regular expression
match = re.search(r'\\(\d+\.\d+v\d+)\.pdf$', file_path)

if match:
    id_with_version = match.group(1)
    print("Extracted ID with version:", id_with_version)
else:
    print("No match found.")

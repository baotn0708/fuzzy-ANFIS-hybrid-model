
import PyPDF2
import os

def read_pdf(file_path, num_pages=10):
    print(f"\n{'='*50}")
    print(f"Reading {os.path.basename(file_path)}")
    print(f"{'='*50}")
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            max_pages = min(len(reader.pages), num_pages)
            for i in range(max_pages):
                page = reader.pages[i]
                text = page.extract_text()
                print(f"\n--- Page {i+1} ---\n")
                print(text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

files = ['/Users/bao/Documents/tsa_paper_1/ANFIS.pdf', '/Users/bao/Documents/tsa_paper_1/peerj-cs-3004.pdf']
for f in files:
    read_pdf(f)

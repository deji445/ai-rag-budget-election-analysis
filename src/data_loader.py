import pandas as pd
from PyPDF2 import PdfReader

# Load CSV
def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna()  # simple cleaning
    return df

# Load PDF
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text
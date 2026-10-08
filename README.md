# NC-PDFparserandextractor

A scientific paper parsing and keyword sentence extraction tool built with **Python**, **GROBID**, **NLTK**, and **Streamlit**.

## Features

- **Full-Text Parsing via GROBID**: Upload academic PDF papers to extract TEI-XML structure.
- **Selective Section Extraction**: Extracts targeted academic sections:
  - Abstract
  - Introduction
  - Result and Discussion
  - Conclusion
  *(Safely ignores sections that are not present)*
- **NLTK Sentence Tokenization**: Uses NLTK `punkt_tab` with scientific abbreviation handling (e.g., `equiv.`, `ref.`) to accurately split text into sentences without false splits.
- **Case-Insensitive Keyword Mining**: Filters and highlights sentences containing specific keywords (default: *syringic acid*).
- **Interactive Streamlit Web App**:
  - Drag-and-drop PDF or upload existing TEI-XML
  - Live keyword highlighting (`<mark>`)
  - Full-text section reader
  - One-click export to Markdown (`.md`), CSV (`.csv`), or raw TEI-XML (`.xml`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/nuttagritc-pog/NC-PDFparserandextractor.git
   cd NC-PDFparserandextractor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web App

Start the Streamlit application:
```bash
python -m streamlit run app.py
```
Or on Windows, simply double-click `run_app.bat`.

Then open your browser at `http://localhost:8501`.

## CLI Usage

### 1. Extract Target Sections from TEI-XML
```bash
python extract_sections.py molbank-2025-M2060.tei.xml
```

### 2. Sentence Tokenization & Keyword Filtering
```bash
python tokenize_sections.py molbank-2025-M2060.tei.xml
```

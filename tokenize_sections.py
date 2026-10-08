#!/usr/bin/env python3
"""
tokenize_sections.py

Uses NLTK sentence tokenization (with punkt_tab) to extract sentences from:
  - Abstract
  - Introduction
  - Result and Discussion
  - Conclusion

Filters sentences so that each sentence MUST contain the keyword "syringic acid"
(treated as case-insensitive).
"""

import sys
import re
from pathlib import Path
import nltk

# Ensure punkt_tab tokenizer model is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize
from extract_sections import extract_sections


KEYWORD = "syringic acid"


# Extra scientific abbreviations to prevent false splits on dots
SCIENTIFIC_ABBREVS = {'equiv', 'fig', 'ref', 'ca', 'approx', 'no', 'vol', 'scheme'}


def filter_sentences_by_keyword(text: str, keyword: str = KEYWORD) -> list[str]:
    """
    Tokenizes text into sentences using NLTK (punkt_tab) and retains only those
    containing the keyword (case-insensitive).
    """
    if not text:
        return []

    # Load English punkt_tab tokenizer and register domain abbreviations
    tokenizer = nltk.data.load('tokenizers/punkt_tab/english.pickle')
    tokenizer._params.abbrev_types.update(SCIENTIFIC_ABBREVS)

    sentences = tokenizer.tokenize(text)

    # Case-insensitive keyword matching
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matched = [s.strip() for s in sentences if pattern.search(s)]

    return matched



def process_document(xml_path: str, keyword: str = KEYWORD):
    """
    Extracts sections and filters sentences by case-insensitive keyword.
    """
    sections = extract_sections(xml_path)
    
    results = {}
    for section_name in ['Abstract', 'Introduction', 'Result and Discussion', 'Conclusion']:
        if section_name in sections and sections[section_name]:
            matched_sentences = filter_sentences_by_keyword(sections[section_name], keyword)
            results[section_name] = matched_sentences

    return results


def main():
    xml_file = sys.argv[1] if len(sys.argv) > 1 else 'molbank-2025-M2060.tei.xml'
    
    print(f"File: {xml_file}")
    print(f"Keyword (case-insensitive): '{KEYWORD}'")
    print("=" * 70)

    results = process_document(xml_file, KEYWORD)

    output_lines = [
        f"# Sentences Containing '{KEYWORD}' (Case-Insensitive)\n",
        f"**Source Document**: `{xml_file}`\n"
    ]

    total_matches = 0
    for section_name, sentences in results.items():
        print(f"\n[{section_name}] ({len(sentences)} sentence(s) matched):")
        output_lines.append(f"## {section_name} ({len(sentences)} matched)\n")

        if not sentences:
            print("  (No sentences matched the keyword)")
            output_lines.append("*(No sentences matched)*\n")
            continue

        for idx, sentence in enumerate(sentences, 1):
            clean_sentence = ' '.join(sentence.split())
            print(f"  {idx}. {clean_sentence}")
            output_lines.append(f"{idx}. {clean_sentence}\n")
            total_matches += 1

    print("\n" + "=" * 70)
    print(f"Total matching sentences found: {total_matches}")

    # Save output to Markdown file
    output_path = Path(xml_file).with_suffix('.syringic_acid_sentences.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"Saved output to: {output_path}")


if __name__ == '__main__':
    main()

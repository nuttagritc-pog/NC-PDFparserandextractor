#!/usr/bin/env python3
"""
extract_sections.py

Extracts text from specified sections of a GROBID TEI-XML file:
  - Abstract
  - Introduction
  - Result and Discussion
  - Conclusion

If a tag or section cannot be found, it is ignored and the script moves on to the next.
"""

import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def strip_namespace(tag: str) -> str:
    """Removes XML namespace prefix if present, e.g. {http://...}div -> div."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def get_clean_text(element: ET.Element) -> str:
    """Extracts and cleans all text from an XML element, preserving paragraph flow."""
    paragraphs = []
    # Find all paragraph tags or iterate over text
    p_elements = [child for child in element.iter() if strip_namespace(child.tag) == 'p']
    if p_elements:
        for p in p_elements:
            text = ' '.join(''.join(p.itertext()).split())
            if text:
                paragraphs.append(text)
    else:
        # Fallback to direct text if no <p> tags
        text = ' '.join(''.join(element.itertext()).split())
        if text:
            paragraphs.append(text)
    return '\n\n'.join(paragraphs)


def find_abstract(root: ET.Element) -> str | None:
    """Looks for the abstract tag or abstract section."""
    # 1. Direct tag search for <abstract> anywhere in tree
    for elem in root.iter():
        if strip_namespace(elem.tag).lower() == 'abstract':
            text = get_clean_text(elem)
            if text:
                return text

    # 2. Look for div with type='abstract' or head containing 'abstract'
    for elem in root.iter():
        if strip_namespace(elem.tag).lower() == 'div':
            div_type = elem.attrib.get('type', '').lower()
            if div_type == 'abstract':
                text = get_clean_text(elem)
                if text:
                    return text
            for child in elem:
                if strip_namespace(child.tag).lower() == 'head':
                    head_txt = ''.join(child.itertext()).strip().lower()
                    if 'abstract' in head_txt:
                        text = get_clean_text(elem)
                        if text:
                            return text
    return None


def find_body_sections(root: ET.Element) -> dict[str, str]:
    """
    Parses body divs to match sections:
      - Introduction
      - Result and Discussion
      - Conclusion
    Handles both direct tag names (<introduction>, etc.) and GROBID TEI <div> structures.
    """
    results = {}

    # Target categories and their matching patterns
    patterns = {
        'Introduction': [r'\bintro(?:duction)?\b'],
        'Result and Discussion': [
            r'\bresults?\s+and\s+discussion\b',
            r'\bresults?\s*&\s*discussion\b',
            r'\bresults?\b',
            r'\bdiscussion\b'
        ],
        'Conclusion': [r'\bconclu(?:sion|sions)?\b']
    }

    # First check: Are there explicit tags named <introduction>, <results>, etc.?
    for elem in root.iter():
        tag = strip_namespace(elem.tag).lower()
        for sec_name, sec_patterns in patterns.items():
            if sec_name not in results:
                for pat in sec_patterns:
                    if re.search(pat, tag, re.IGNORECASE):
                        txt = get_clean_text(elem)
                        if txt:
                            results[sec_name] = txt
                            break

    # Second check: TEI <body> <div> elements with <head> or @type
    # Gather all <div> elements inside <body>
    body_divs = []
    for elem in root.iter():
        if strip_namespace(elem.tag).lower() == 'body':
            for div in elem.iter():
                if strip_namespace(div.tag).lower() == 'div':
                    head_elem = None
                    for child in div:
                        if strip_namespace(child.tag).lower() == 'head':
                            head_elem = child
                            break
                    head_text = ''.join(head_elem.itertext()).strip() if head_elem is not None else ''
                    head_n = head_elem.attrib.get('n', '') if head_elem is not None else ''
                    div_type = div.attrib.get('type', '')
                    body_divs.append({
                        'div': div,
                        'head': head_text,
                        'n': head_n,
                        'type': div_type
                    })

    # Group or match divs
    # E.g. in GROBID, section 2 may have head "2. Results and Discussion" and subsections "2.1.", "2.2."
    for i, item in enumerate(body_divs):
        combined_heading = f"{item['n']} {item['head']} {item['type']}".lower()

        # Check Introduction
        if 'Introduction' not in results:
            if re.search(r'\bintro(?:duction)?\b', combined_heading):
                txt = get_clean_text(item['div'])
                if txt:
                    results['Introduction'] = txt

        # Check Conclusion
        if 'Conclusion' not in results:
            if re.search(r'\bconclu(?:sion|sions)?\b', combined_heading):
                txt = get_clean_text(item['div'])
                if txt:
                    results['Conclusion'] = txt

    # For Result and Discussion: collect main section and any subsections (e.g. 2, 2.1, 2.2)
    if 'Result and Discussion' not in results:
        res_texts = []
        in_res_section = False
        main_section_prefix = None

        for item in body_divs:
            head_txt = item['head'].lower()
            n_txt = item['n'].strip()
            
            # Check if this is the start of Results and Discussion
            if re.search(r'\bresults?\s*(?:and|&)\s*discussion\b', head_txt) or \
               (re.search(r'\bresults?\b', head_txt) and not in_res_section):
                in_res_section = True
                main_section_prefix = n_txt.split('.')[0] if '.' in n_txt or n_txt.isdigit() else None
                txt = get_clean_text(item['div'])
                if txt:
                    heading_prefix = f"### {item['head']}\n" if item['head'] else ""
                    res_texts.append(f"{heading_prefix}{txt}")
                continue

            if in_res_section:
                # Check if still in subsection (e.g. 2.1, 2.2 when main was 2)
                if main_section_prefix and n_txt.startswith(main_section_prefix + '.'):
                    txt = get_clean_text(item['div'])
                    if txt:
                        heading_prefix = f"### {item['head']}\n" if item['head'] else ""
                        res_texts.append(f"{heading_prefix}{txt}")
                else:
                    # New top-level section started (e.g. 3. Materials and Methods)
                    if n_txt and not n_txt.startswith(main_section_prefix or ''):
                        break

        if res_texts:
            results['Result and Discussion'] = '\n\n'.join(res_texts)

    return results


def extract_sections(xml_path: str) -> dict[str, str]:
    """
    Extracts abstract, introduction, result and discussion, and conclusion tags.
    If a tag/section cannot be found, it is ignored.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    extracted = {}

    # 1. Abstract
    abstract_text = find_abstract(root)
    if abstract_text:
        extracted['Abstract'] = abstract_text

    # 2, 3, 4. Body sections
    body_sections = find_body_sections(root)
    
    # Target order
    target_sections = ['Introduction', 'Result and Discussion', 'Conclusion']
    for sec in target_sections:
        if sec in body_sections and body_sections[sec]:
            extracted[sec] = body_sections[sec]

    return extracted


def main():
    xml_file = sys.argv[1] if len(sys.argv) > 1 else 'molbank-2025-M2060.tei.xml'
    
    print(f"Parsing: {xml_file}")
    sections = extract_sections(xml_file)

    if not sections:
        print("No matching sections found.")
        return

    output_lines = []
    print("\n" + "=" * 60)
    for title, content in sections.items():
        header = f"=== {title.upper()} ==="
        print(header)
        print(content)
        print("=" * 60 + "\n")

        output_lines.append(f"# {title}\n\n{content}\n")

    # Also save to an output markdown file
    output_path = Path(xml_file).with_suffix('.extracted.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(output_lines))
    print(f"Saved extracted sections to: {output_path}")


if __name__ == '__main__':
    main()

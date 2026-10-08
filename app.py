#!/usr/bin/env python3
"""
app.py - GROBID Paper Section & Sentence Extractor Web App
Built with Streamlit and NLTK.
"""

import io
import re
import xml.etree.ElementTree as ET
import pandas as pd
import requests
import streamlit as st
import nltk

# Ensure punkt_tab tokenizer is ready
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Register common scientific abbreviations
SCIENTIFIC_ABBREVS = {"equiv", "fig", "ref", "ca", "approx", "no", "vol", "scheme", "dept"}


# -------------------------------------------------------------
# Core Extraction & Tokenization Functions
# -------------------------------------------------------------
def strip_namespace(tag: str) -> str:
    """Removes XML namespace prefix if present, e.g. {http://...}div -> div."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def get_clean_text(element: ET.Element) -> str:
    """Extracts and cleans all text from an XML element, preserving paragraph flow."""
    paragraphs = []
    p_elements = [child for child in element.iter() if strip_namespace(child.tag) == "p"]
    if p_elements:
        for p in p_elements:
            text = " ".join("".join(p.itertext()).split())
            if text:
                paragraphs.append(text)
    else:
        text = " ".join("".join(element.itertext()).split())
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def extract_metadata(root: ET.Element) -> dict:
    """Extracts paper title and authors from TEI XML."""
    title_elem = None
    for elem in root.iter():
        if strip_namespace(elem.tag) == "title" and elem.attrib.get("type") == "main":
            title_elem = elem
            break
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else "Untitled Paper"

    authors = []
    for author in root.iter():
        if strip_namespace(author.tag) == "author":
            name_parts = [n.text for n in author.iter() if strip_namespace(n.tag) in ("forename", "surname") and n.text]
            if name_parts:
                authors.append(" ".join(name_parts))
    return {"title": title, "authors": list(dict.fromkeys(authors))}


def find_abstract(root: ET.Element) -> str | None:
    """Finds the abstract text in TEI XML."""
    for elem in root.iter():
        if strip_namespace(elem.tag).lower() == "abstract":
            text = get_clean_text(elem)
            if text:
                return text

    for elem in root.iter():
        if strip_namespace(elem.tag).lower() == "div":
            div_type = elem.attrib.get("type", "").lower()
            if div_type == "abstract":
                text = get_clean_text(elem)
                if text:
                    return text
            for child in elem:
                if strip_namespace(child.tag).lower() == "head":
                    head_txt = "".join(child.itertext()).strip().lower()
                    if "abstract" in head_txt:
                        text = get_clean_text(elem)
                        if text:
                            return text
    return None


def find_body_sections(root: ET.Element) -> dict[str, str]:
    """Finds Introduction, Result and Discussion, and Conclusion sections."""
    results = {}
    body_divs = []
    for elem in root.iter():
        if strip_namespace(elem.tag).lower() == "body":
            for div in elem.iter():
                if strip_namespace(div.tag).lower() == "div":
                    head_elem = None
                    for child in div:
                        if strip_namespace(child.tag).lower() == "head":
                            head_elem = child
                            break
                    head_text = "".join(head_elem.itertext()).strip() if head_elem is not None else ""
                    head_n = head_elem.attrib.get("n", "") if head_elem is not None else ""
                    body_divs.append({"div": div, "head": head_text, "n": head_n})

    for item in body_divs:
        heading = f"{item['n']} {item['head']}".lower()
        if "Introduction" not in results and re.search(r"\bintro(?:duction)?\b", heading):
            txt = get_clean_text(item["div"])
            if txt:
                results["Introduction"] = txt
        if "Conclusion" not in results and re.search(r"\bconclu(?:sion|sions)?\b", heading):
            txt = get_clean_text(item["div"])
            if txt:
                results["Conclusion"] = txt

    # Collect Results and Discussion
    if "Result and Discussion" not in results:
        res_texts = []
        in_res = False
        main_prefix = None
        for item in body_divs:
            head_txt = item["head"].lower()
            n_txt = item["n"].strip()
            if re.search(r"\bresults?\s*(?:and|&)\s*discussion\b", head_txt) or (re.search(r"\bresults?\b", head_txt) and not in_res):
                in_res = True
                main_prefix = n_txt.split(".")[0] if "." in n_txt or n_txt.isdigit() else None
                txt = get_clean_text(item["div"])
                if txt:
                    sub_h = f"**{item['head']}**\n\n" if item["head"] else ""
                    res_texts.append(f"{sub_h}{txt}")
                continue
            if in_res:
                if main_prefix and n_txt.startswith(main_prefix + "."):
                    txt = get_clean_text(item["div"])
                    if txt:
                        sub_h = f"**{item['head']}**\n\n" if item["head"] else ""
                        res_texts.append(f"{sub_h}{txt}")
                else:
                    if n_txt and not n_txt.startswith(main_prefix or ""):
                        break
        if res_texts:
            results["Result and Discussion"] = "\n\n".join(res_texts)

    return results


def parse_tei_xml(xml_content: str | bytes) -> tuple[dict, dict[str, str]]:
    """Parses XML and returns (metadata, sections_dict)."""
    if isinstance(xml_content, str):
        root = ET.fromstring(xml_content)
    else:
        root = ET.fromstring(xml_content.decode("utf-8", errors="ignore"))

    meta = extract_metadata(root)
    sections = {}

    abstract = find_abstract(root)
    if abstract:
        sections["Abstract"] = abstract

    body_secs = find_body_sections(root)
    for k in ["Introduction", "Result and Discussion", "Conclusion"]:
        if k in body_secs and body_secs[k]:
            sections[k] = body_secs[k]

    return meta, sections


def tokenize_and_filter(text: str, keyword: str, case_insensitive: bool = True) -> list[str]:
    """Tokenizes text using NLTK and filters sentences by keyword."""
    if not text or not keyword:
        return []

    tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
    tokenizer._params.abbrev_types.update(SCIENTIFIC_ABBREVS)

    sentences = tokenizer.tokenize(text)

    flags = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(re.escape(keyword.strip()), flags)

    return [s.strip() for s in sentences if pattern.search(s)]


def highlight_keyword(text: str, keyword: str, case_insensitive: bool = True) -> str:
    """Wraps matches of keyword in an HTML <mark> tag."""
    flags = re.IGNORECASE if case_insensitive else 0
    pattern = re.compile(f"({re.escape(keyword.strip())})", flags)
    return pattern.sub(r"<mark style='background-color: #fef08a; padding: 2px 4px; border-radius: 4px; font-weight: bold;'>\1</mark>", text)


def process_pdf_with_grobid(pdf_bytes: bytes, grobid_url: str) -> str:
    """Calls GROBID processFulltextDocument API."""
    endpoint = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"
    files = {"input": ("document.pdf", pdf_bytes, "application/pdf")}
    data = {"consolidateHeader": "1", "consolidateCitations": "1"}
    
    response = requests.post(endpoint, files=files, data=data, timeout=180)
    response.raise_for_status()
    return response.text


# -------------------------------------------------------------
# Streamlit App UI
# -------------------------------------------------------------
st.set_page_config(
    page_title="PDF Section & Keyword Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Scientific Paper Keyword & Section Extractor")
st.markdown(
    "Extract targeted sections (**Abstract, Introduction, Results & Discussion, Conclusion**) "
    "from PDF papers via **GROBID**, then tokenize and mine sentences with keyword filtering via **NLTK**."
)

# Sidebar configurations
with st.sidebar:
    st.header("⚙️ Configuration")
    grobid_url = st.text_input(
        "GROBID Service URL",
        value="https://grobidorg-grobid.hf.space",
        help="Use Hugging Face demo or your local Docker instance (e.g., http://localhost:8070)",
    )

    st.subheader("🔍 Keyword Filter")
    keyword_input = st.text_input("Target Keyword", value="syringic acid")
    case_insensitive = st.checkbox("Case-insensitive match", value=True)

    st.subheader("📑 Target Sections")
    selected_sections = st.multiselect(
        "Sections to include:",
        ["Abstract", "Introduction", "Result and Discussion", "Conclusion"],
        default=["Abstract", "Introduction", "Result and Discussion", "Conclusion"],
    )

    st.divider()
    st.info(
        "💡 **Tip**: If you already have the processed TEI-XML file, you can upload it directly to skip the GROBID API step."
    )

# Input Tabs: Upload PDF or Upload TEI-XML
input_tab1, input_tab2 = st.tabs(["📤 Upload PDF Document", "📋 Upload Existing TEI-XML"])

xml_source = None
filename_base = "document"

with input_tab1:
    uploaded_pdf = st.file_uploader("Choose a PDF research paper", type=["pdf"])
    col_pdf1, col_pdf2 = st.columns([1, 4])
    with col_pdf1:
        load_sample = st.button("Use Sample PDF (molbank-2025-M2060.pdf)")

    if load_sample:
        try:
            with open("molbank-2025-M2060.pdf", "rb") as f:
                uploaded_pdf = io.BytesIO(f.read())
                uploaded_pdf.name = "molbank-2025-M2060.pdf"
            st.success("Loaded sample file: `molbank-2025-M2060.pdf`")
        except FileNotFoundError:
            st.error("Sample PDF not found in workspace.")

with input_tab2:
    uploaded_xml = st.file_uploader("Choose a TEI XML file", type=["xml"])
    if st.button("Use Generated Sample TEI XML (molbank-2025-M2060.tei.xml)"):
        try:
            with open("molbank-2025-M2060.tei.xml", "r", encoding="utf-8") as f:
                xml_source = f.read()
                filename_base = "molbank-2025-M2060"
            st.success("Loaded sample XML: `molbank-2025-M2060.tei.xml`")
        except FileNotFoundError:
            st.error("Sample XML not found.")

# Trigger processing
if uploaded_xml is not None and xml_source is None:
    xml_source = uploaded_xml.read().decode("utf-8", errors="ignore")
    filename_base = uploaded_xml.name.rsplit(".", 1)[0]

process_clicked = st.button("🚀 Process & Extract Sentences", type="primary", use_container_width=True)

if process_clicked:
    if not uploaded_pdf and not xml_source:
        st.warning("Please upload a PDF file or TEI-XML file first.")
    else:
        # Step 1: GROBID conversion if PDF uploaded
        if xml_source is None and uploaded_pdf is not None:
            filename_base = getattr(uploaded_pdf, "name", "document").rsplit(".", 1)[0]
            with st.spinner("Step 1/3: Sending PDF to GROBID server for TEI-XML parsing..."):
                try:
                    pdf_bytes = uploaded_pdf.read() if hasattr(uploaded_pdf, "read") else uploaded_pdf.getvalue()
                    xml_source = process_pdf_with_grobid(pdf_bytes, grobid_url)
                    st.success("GROBID extraction complete!")
                except Exception as e:
                    st.error(f"Error communicating with GROBID: {e}")
                    st.stop()

        # Step 2: Parse sections
        with st.spinner("Step 2/3: Parsing target sections from TEI-XML..."):
            try:
                meta, sections = parse_tei_xml(xml_source)
            except Exception as e:
                st.error(f"Failed to parse TEI-XML: {e}")
                st.stop()

        # Step 3: Tokenize & Filter
        with st.spinner("Step 3/3: Running NLTK sentence tokenization & keyword filtering..."):
            extracted_results = {}
            flat_rows = []
            for sec_name in selected_sections:
                if sec_name in sections:
                    matched_sents = tokenize_and_filter(sections[sec_name], keyword_input, case_insensitive)
                    extracted_results[sec_name] = matched_sents
                    for idx, s in enumerate(matched_sents, 1):
                        flat_rows.append({"Section": sec_name, "Index": idx, "Sentence": s})
                else:
                    extracted_results[sec_name] = []

        # Store in session state for interactive viewing
        st.session_state["results"] = extracted_results
        st.session_state["sections"] = sections
        st.session_state["meta"] = meta
        st.session_state["flat_rows"] = flat_rows
        st.session_state["filename_base"] = filename_base
        st.session_state["xml_source"] = xml_source

# Display results if present in session state
if "results" in st.session_state:
    results = st.session_state["results"]
    sections = st.session_state["sections"]
    meta = st.session_state["meta"]
    flat_rows = st.session_state["flat_rows"]
    filename_base = st.session_state["filename_base"]

    st.divider()

    # Paper Metadata Header
    st.subheader(f"📑 {meta.get('title', 'Extracted Document')}")
    if meta.get("authors"):
        st.caption(f"**Authors**: {', '.join(meta['authors'])}")

    # Summary Metrics
    total_matches = len(flat_rows)
    cols = st.columns(len(selected_sections) + 1)
    cols[0].metric("Total Matches", f"{total_matches} sentences")
    for i, sec_name in enumerate(selected_sections):
        count = len(results.get(sec_name, []))
        cols[i + 1].metric(sec_name, f"{count} sents")

    # Tabs for each Section
    st.write("---")
    sec_tabs = st.tabs([f"{sec} ({len(results.get(sec, []))})" for sec in selected_sections])

    for tab, sec_name in zip(sec_tabs, selected_sections):
        with tab:
            matched_sents = results.get(sec_name, [])
            if not matched_sents:
                if sec_name not in sections:
                    st.info(f"Tag or section `{sec_name}` was not found in the document (safely ignored).")
                else:
                    st.warning(f"Section `{sec_name}` is present, but contains 0 sentences with keyword '{keyword_input}'.")
            else:
                for idx, sent in enumerate(matched_sents, 1):
                    highlighted = highlight_keyword(sent, keyword_input, case_insensitive)
                    st.markdown(
                        f"""
                        <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                            <span style="color: #64748b; font-size: 0.85em; font-weight: bold;">#{idx}</span><br/>
                            <span style="font-size: 1.02em; color: #1e293b;">{highlighted}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Collapsible Full Section Text
            if sec_name in sections:
                with st.expander(f"📖 View Full Text of {sec_name}"):
                    st.markdown(sections[sec_name])

    # Download Buttons
    st.divider()
    st.subheader("💾 Export Extracted Results")

    col_d1, col_d2, col_d3 = st.columns(3)

    # 1. Markdown Export
    md_content = [f"# Sentences Matching '{keyword_input}' in {meta['title']}\n"]
    for sec_name, sents in results.items():
        md_content.append(f"\n## {sec_name} ({len(sents)} matched)\n")
        if not sents:
            md_content.append("*(No matching sentences)*\n")
        for i, s in enumerate(sents, 1):
            md_content.append(f"{i}. {s}\n")
    md_text = "\n".join(md_content)

    with col_d1:
        st.download_button(
            "📥 Download as Markdown (.md)",
            data=md_text,
            file_name=f"{filename_base}_extracted_sentences.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # 2. CSV Export
    if flat_rows:
        df = pd.DataFrame(flat_rows)
        csv_data = df.to_csv(index=False).encode("utf-8")
        with col_d2:
            st.download_button(
                "📥 Download as CSV (.csv)",
                data=csv_data,
                file_name=f"{filename_base}_extracted_sentences.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 3. Raw TEI-XML Export
    if "xml_source" in st.session_state and st.session_state["xml_source"]:
        with col_d3:
            st.download_button(
                "📥 Download Raw TEI-XML (.xml)",
                data=st.session_state["xml_source"],
                file_name=f"{filename_base}.tei.xml",
                mime="application/xml",
                use_container_width=True,
            )

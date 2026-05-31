import os
import re
import datetime
from flask import Flask, render_template, request, jsonify, session, send_file
import fitz  # PyMuPDF
from groq import Groq
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)
# A secret key is required to secure Flask sessions (client-side encrypted cookies)
app.secret_key = "research_assistant_secret_session_key_1337"

# Configure directories and upload limits
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed extensions check
def allowed_file(filename):
    """
    Checks if the uploaded file has a PDF extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

# Initialize Groq client
# Reads from environment variables or uses a placeholder.
# Set your environment variable GROQ_API_KEY, or replace "YOUR_GROQ_API_KEY" with your actual key.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

def get_groq_client():
    """
    Helper function to initialize and return the Groq API client.
    """
    return Groq(api_key=GROQ_API_KEY)

def extract_text_from_pdf(pdf_path):
    """
    Extracts plain text from a PDF file using PyMuPDF (fitz).
    Returns the extracted text as a single string.
    """
    text = ""
    # Open the PDF document
    doc = fitz.open(pdf_path)
    # Loop through each page and extract text
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def get_word_count(text):
    """
    Calculates the word count of a given string.
    """
    return len(text.split())

def perform_smart_chunking(text, query, chunk_size=3000, overlap=500):
    """
    Splits text into overlapping chunks of words if word count exceeds 6000 words.
    Then, performs a lightweight BM25-like lexical overlap ranking between the 
    user's query and each chunk to find and return the most relevant chunk.
    """
    words = text.split()
    total_words = len(words)
    
    # If the text is small enough, no chunking is needed
    if total_words <= 6000:
        return text, False

    # Create overlapping chunks
    chunks = []
    start = 0
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == total_words:
            break
        start += (chunk_size - overlap)

    # Simple Stopwords to filter out from query for lexical matching
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'what',
        'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'i', 'me', 'my', 'we', 'our'
    }

    # Clean the query: lowercase, strip punctuation, and filter out stopwords
    cleaned_query = re.sub(r'[^\w\s]', '', query.lower())
    query_words = [w for w in cleaned_query.split() if w not in stopwords]

    # If the query is empty after filtering, default to matching all query terms
    if not query_words:
        query_words = cleaned_query.split()

    # Rank chunks based on frequency match of query keywords
    best_chunk_idx = 0
    max_score = -1

    for idx, chunk in enumerate(chunks):
        score = 0
        chunk_lower = chunk.lower()
        # Calculate overlap score
        for word in query_words:
            # Count occurrences of the word in this chunk
            score += chunk_lower.count(word)
        
        if score > max_score:
            max_score = score
            best_chunk_idx = idx

    # Return the most relevant chunk and a True flag indicating chunking was applied
    return chunks[best_chunk_idx], True

@app.route('/')
def home():
    """
    Renders the main dashboard page.
    Initializes session variables if they don't exist.
    """
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'uploaded_files' not in session:
        session['uploaded_files'] = {}  # {filename: {"path": path, "word_count": wc}}
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Handles multiple PDF uploads (up to 3 files).
    Performs size checks, PDF type checks, text extraction, and empty PDF detection.
    Stores metadata in the session and returns file statistics to the client.
    """
    # Initialize uploaded files in session if missing
    if 'uploaded_files' not in session:
        session['uploaded_files'] = {}
        
    uploaded_dict = session['uploaded_files']

    # Enforce a maximum of 3 uploaded files
    if len(uploaded_dict) >= 3:
        # Check if the user is attempting to upload even more files
        if 'files[]' in request.files:
            return jsonify({"error": "You can upload a maximum of 3 PDFs. Please clear existing files to upload new ones."}), 400

    if 'files[]' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    files = request.files.getlist('files[]')
    
    # Check if empty files
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected for upload."}), 400

    # Ensure cumulative or new upload doesn't exceed 3 files
    new_files_count = len(files)
    if len(uploaded_dict) + new_files_count > 3:
        return jsonify({"error": f"Uploading these files would exceed the 3 PDF limit. Currently uploaded: {len(uploaded_dict)}."}), 400

    response_data = []

    for file in files:
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format. Please upload PDF files only."}), 400

        # Secure and sanitize the file name
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file to disk
        file.save(save_path)

        try:
            # Extract plain text from PDF using PyMuPDF
            extracted_text = extract_text_from_pdf(save_path)
            word_count = get_word_count(extracted_text)

            # Check if document has no extractable text (e.g. fully scanned image-based PDF)
            if word_count == 0:
                # Remove unreadable file from disk
                if os.path.exists(save_path):
                    os.remove(save_path)
                return jsonify({"error": f"'{filename}' appears to be scanned or empty. Please use a text-based PDF."}), 400

            # Store text in a companion txt file for extremely stable backend retrieval (prevents session size limits)
            text_save_path = save_path + ".txt"
            with open(text_save_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            # Add to session tracking
            uploaded_dict[filename] = {
                "path": save_path,
                "text_path": text_save_path,
                "word_count": word_count
            }
            
            response_data.append({
                "filename": filename,
                "word_count": word_count
            })

        except Exception as e:
            # Clean up on extraction error
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({"error": f"Failed to process '{filename}': {str(e)}"}), 500

    # Update Flask session
    session['uploaded_files'] = uploaded_dict
    session.modified = True

    return jsonify({"success": True, "files": response_data})

@app.route('/summarize', methods=['POST'])
def summarize_document():
    """
    Generates a highly structured auto-summary of the selected PDF.
    Retrieves full document text (or smart-chunks if large) and prompts the Groq API.
    """
    data = request.json
    filename = data.get('filename')

    if not filename or 'uploaded_files' not in session or filename not in session['uploaded_files']:
        return jsonify({"error": "Selected document is missing or invalid."}), 400

    file_info = session['uploaded_files'][filename]
    text_path = file_info.get('text_path')

    if not text_path or not os.path.exists(text_path):
        return jsonify({"error": "Document text could not be loaded."}), 404

    # Read extracted text
    with open(text_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()

    # Smart-chunking for summary if extremely massive, though we'll try to feed a substantial summary segment
    # Let's extract the first 6000 words for the summary to capture Abstract, Intro, Methodology, and Main Results
    words = doc_text.split()
    is_chunked = False
    if len(words) > 6000:
        doc_text = " ".join(words[:6000])
        is_chunked = True

    # Prompt template for structure summary extraction
    summary_prompt = (
        "Analyze the following research document text and generate a structured summary. "
        "Your response MUST strictly follow this exact format with clean headings:\n\n"
        "### ONE-LINE SUMMARY\n"
        "[Provide a concise, impact-focused single sentence summary of the paper's core objective.]\n\n"
        "### KEY CONTRIBUTIONS\n"
        "[List 3-4 major contributions or innovations introduced by this work in bullet points.]\n\n"
        "### METHODOLOGY IN SIMPLE WORDS\n"
        "[Explain the methodology, framework, or experimental design in intuitive, simple language suitable for non-experts.]\n\n"
        "### MAIN RESULTS & FINDINGS\n"
        "[Describe the primary results, metrics, benchmarks, or key conclusions achieved by the authors.]\n\n"
        f"Document Content:\n{doc_text}"
    )

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert research paper assistant. Answer questions clearly, cite sections, explain technical terms in simple language. If the answer is not in the document, say 'This paper does not cover that.' Never guess or make up information."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=1024
        )
        summary_result = response.choices[0].message.content

        return jsonify({
            "success": True,
            "summary": summary_result,
            "is_chunked": is_chunked
        })

    except Exception as e:
        # Graceful Groq API fail error handling
        return jsonify({"error": "AI service is temporarily unavailable. Please try again."}), 500

@app.route('/query', methods=['POST'])
def query_document():
    """
    Handles user Q&A queries.
    Utilizes smart-chunking to retrieve the most relevant sections of text for papers > 6000 words.
    Constructs the prompt, queries Groq API, and appends transaction to Flask session chat history.
    """
    data = request.json
    filename = data.get('filename')
    question = data.get('question')

    if not filename or not question:
        return jsonify({"error": "Filename and question are required."}), 400

    if 'uploaded_files' not in session or filename not in session['uploaded_files']:
        return jsonify({"error": "Selected document is not loaded in current session."}), 400

    file_info = session['uploaded_files'][filename]
    text_path = file_info.get('text_path')

    if not text_path or not os.path.exists(text_path):
        return jsonify({"error": "Extracted text file not found on server."}), 404

    # Read extracted text
    with open(text_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Apply smart chunking (6000 words limit threshold)
    relevant_context, is_chunked = perform_smart_chunking(full_text, question, chunk_size=3000, overlap=500)

    # Build standard system prompt + document context
    prompt = (
        f"You are analyzing the research document: '{filename}'.\n"
        f"Use the following text extract from the document to answer the user's question:\n"
        f"--- EXTRACT BEGINS ---\n{relevant_context}\n--- EXTRACT ENDS ---\n\n"
        f"User Question: {question}"
    )

    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert research paper assistant. Answer questions clearly, cite sections, explain technical terms in simple language. If the answer is not in the document, say 'This paper does not cover that.' Never guess or make up information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024
        )
        answer = response.choices[0].message.content

        # Retrieve, update, and save to Flask session chat history
        chat_history = session.get('chat_history', [])
        chat_history.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pdf_name": filename,
            "question": question,
            "answer": answer,
            "is_chunked": is_chunked
        })
        session['chat_history'] = chat_history
        session.modified = True

        return jsonify({
            "success": True,
            "answer": answer,
            "is_chunked": is_chunked
        })

    except Exception as e:
        # Graceful Groq API fail error handling
        return jsonify({"error": "AI service is temporarily unavailable. Please try again."}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """
    Clears the chat history stored inside the session.
    """
    session['chat_history'] = []
    session.modified = True
    return jsonify({"success": True})

@app.route('/clear_all', methods=['POST'])
def clear_all():
    """
    Clears all uploaded PDFs from disk, deletes cached txt extracts,
    and completely resets the Flask session cache.
    """
    if 'uploaded_files' in session:
        for file_info in session['uploaded_files'].values():
            pdf_path = file_info.get('path')
            txt_path = file_info.get('text_path')
            
            # Remove PDF file
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except:
                    pass
            # Remove companion text extract
            if txt_path and os.path.exists(txt_path):
                try:
                    os.remove(txt_path)
                except:
                    pass

    # Clear all session variables
    session.clear()
    session['chat_history'] = []
    session['uploaded_files'] = {}
    return jsonify({"success": True})

@app.route('/download_history', methods=['GET'])
def download_history():
    """
    Assembles the session's chat history list and formats it into a highly professional
    and clear text report. Downloads this report as a .txt file.
    """
    chat_history = session.get('chat_history', [])
    
    if not chat_history:
        # Create a simple file if history is empty to prevent bad experiences
        content = "AI Research Assistant - Chat History Export\n==========================================\nNo Q&A interactions found in the current session."
    else:
        content = "AI RESEARCH ASSISTANT - EXPORTED CHAT HISTORY\n"
        content += "========================================================\n"
        content += f"Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"Total Exchanges: {len(chat_history)}\n"
        content += "========================================================\n\n"

        for idx, entry in enumerate(chat_history, 1):
            content += f"Exchange #{idx}\n"
            content += f"---------------\n"
            content += f"Document:   {entry.get('pdf_name')}\n"
            content += f"Timestamp:  {entry.get('timestamp')}\n"
            content += f"Question:   {entry.get('question')}\n"
            content += f"Chunked:    {'Yes (Analyzed most relevant section due to document size)' if entry.get('is_chunked') else 'No (Analyzed full document)'}\n"
            content += f"Answer:\n{entry.get('answer')}\n"
            content += "\n" + ("=" * 56) + "\n\n"

    # Save content to a temporary export file inside uploads
    export_path = os.path.join(app.config['UPLOAD_FOLDER'], "chat_history_export.txt")
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(content)

    return send_file(
        export_path,
        as_attachment=True,
        download_name=f"QnA_Chat_History_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mimetype="text/plain"
    )
@app.route('/citation', methods=['POST'])
def generate_citation():
    """
    Extracts citation metadata from the first 2 pages of the active PDF
    using Groq API and formats it into APA and BibTeX citation strings.
    """
    data = request.json
    filename = data.get('filename')

    if not filename or 'uploaded_files' not in session or filename not in session['uploaded_files']:
        return jsonify({"error": "Selected document is missing or invalid."}), 400

    file_info = session['uploaded_files'][filename]
    pdf_path = file_info.get('path')

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({"error": "Document could not be located on server."}), 404

    try:
        # Open PDF and read first 2 pages (contains standard cover/title page metadata)
        doc = fitz.open(pdf_path)
        first_pages_text = ""
        pages_to_read = min(2, len(doc))
        for i in range(pages_to_read):
            first_pages_text += doc[i].get_text()
        doc.close()

        # Prompt Llama-3.3 for structured JSON output
        prompt = (
            "Analyze the following text from the first pages of a research paper and extract the citation metadata. "
            "Your response MUST be ONLY a raw JSON object with the following exact keys: "
            "'title', 'authors' (a list of author names, e.g. ['A. Vaswani', 'N. Shazeer']), "
            "'journal' (or conference name), 'year', 'volume', 'issue', 'pages', 'publisher'. "
            "Do not include any chat formatting, markdown indicators, or extra text. If any field is not found, leave it blank.\n\n"
            f"Paper Text:\n{first_pages_text}"
        )

        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        raw_content = response.choices[0].message.content.strip()
        # Clean markdown wrappers if returned
        if raw_content.startswith("```"):
            if "json" in raw_content[:15]:
                raw_content = raw_content.split("json", 1)[1]
            else:
                raw_content = raw_content.split("\n", 1)[1]
            if raw_content.endswith("```"):
                raw_content = raw_content.rsplit("```", 1)[0]
        raw_content = raw_content.strip()

        # Robust regex-based extraction of the JSON object
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            raw_content = match.group(0)

        import json
        metadata = json.loads(raw_content)

        title = metadata.get('title', filename.replace('.pdf', '').replace('_', ' '))
        authors_list = metadata.get('authors', [])
        journal = metadata.get('journal', '')
        year = metadata.get('year', datetime.datetime.now().strftime("%Y"))
        volume = metadata.get('volume', '')
        issue = metadata.get('issue', '')
        pages = metadata.get('pages', '')
        publisher = metadata.get('publisher', '')

        # Format APA Citation
        if isinstance(authors_list, list) and len(authors_list) > 0:
            if len(authors_list) == 1:
                authors_apa = authors_list[0]
            elif len(authors_list) == 2:
                authors_apa = f"{authors_list[0]} & {authors_list[1]}"
            else:
                authors_apa = f"{authors_list[0]}, et al."
        elif isinstance(authors_list, str):
            authors_apa = authors_list
        else:
            authors_apa = "Unknown Authors"

        apa_citation = f"{authors_apa} ({year}). {title}."
        if journal:
            apa_citation += f" *{journal}*."
        if volume:
            apa_citation += f" {volume}"
            if issue:
                apa_citation += f"({issue})"
        if pages:
            apa_citation += f", {pages}."

        # Format BibTeX Citation
        first_author = "author"
        if isinstance(authors_list, list) and len(authors_list) > 0:
            parts = authors_list[0].split()
            if parts:
                first_author = parts[-1].lower()
        
        cite_key = re.sub(r'\W+', '', first_author) + str(year) + re.sub(r'\W+', '', title.split()[0].lower() if title.split() else 'paper')
        
        if isinstance(authors_list, list):
            authors_bib = " and ".join(authors_list)
        else:
            authors_bib = str(authors_list)

        bibtex_citation = (
            f"@article{{{cite_key},\n"
            f"  title={{{title}}},\n"
            f"  author={{{authors_bib}}},\n"
            f"  year={{{year}}}"
        )
        if journal:
            bibtex_citation += f",\n  journal={{{journal}}}"
        if volume:
            bibtex_citation += f",\n  volume={{{volume}}}"
        if issue:
            bibtex_citation += f",\n  number={{{issue}}}"
        if pages:
            bibtex_citation += f",\n  pages={{{pages}}}"
        if publisher:
            bibtex_citation += f",\n  publisher={{{publisher}}}"
        bibtex_citation += "\n}"

        return jsonify({
            "success": True,
            "apa": apa_citation,
            "bibtex": bibtex_citation
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate citation: {str(e)}"}), 500

@app.route('/flashcards', methods=['POST'])
def generate_flashcards():
    """
    Extracts 6 core technical terms and simple definitions from the active PDF
    and returns them as a JSON list for dynamic 3D flashcard rendering.
    """
    data = request.json
    filename = data.get('filename')

    if not filename or 'uploaded_files' not in session or filename not in session['uploaded_files']:
        return jsonify({"error": "Selected document is missing or invalid."}), 400

    file_info = session['uploaded_files'][filename]
    text_path = file_info.get('text_path')

    if not text_path or not os.path.exists(text_path):
        return jsonify({"error": "Extracted text could not be loaded."}), 404

    # Read extracted text (first 1500 words to get core terms)
    with open(text_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()

    words = doc_text.split()
    snippet = " ".join(words[:1500])

    try:
        prompt = (
            "Analyze the following technical paper extract and extract exactly 6 crucial technical terms, concepts, "
            "architectures, or algorithms along with their simple, easy-to-understand definitions (suitable for active recall study).\n"
            "Your response MUST be ONLY a raw JSON array of objects with exactly two keys: 'term' and 'definition'. "
            "Do not include any chat formatting, markdown blocks, or extra descriptions. "
            "Double-check that the JSON is syntax-correct.\n\n"
            "Example JSON output:\n"
            "[\n"
            "  {\"term\": \"Transformer\", \"definition\": \"A deep learning model architecture that uses attention mechanisms to process sequences in parallel.\"}\n"
            "]\n\n"
            f"Paper text:\n{snippet}"
        )

        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )

        raw_content = response.choices[0].message.content.strip()
        # Clean markdown wrappers if returned
        if raw_content.startswith("```"):
            if "json" in raw_content[:15]:
                raw_content = raw_content.split("json", 1)[1]
            else:
                raw_content = raw_content.split("\n", 1)[1]
            if raw_content.endswith("```"):
                raw_content = raw_content.rsplit("```", 1)[0]
        raw_content = raw_content.strip()

        # Robust regex-based extraction of the JSON array
        match = re.search(r'\[.*\]', raw_content, re.DOTALL)
        if match:
            raw_content = match.group(0)

        import json
        flashcards_list = json.loads(raw_content)

        return jsonify({
            "success": True,
            "flashcards": flashcards_list
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate study flashcards: {str(e)}"}), 500

if __name__ == '__main__':
    # Starts the local development server on port 5000 as requested
    app.run(host='localhost', port=5000, debug=True)

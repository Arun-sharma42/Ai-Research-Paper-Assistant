# AI Research Paper & Technical Document Understanding Assistant

A robust, premium-grade technical document understanding assistant built using **Python, Flask, and the Groq API** (`llama-3.3-70b-versatile` model). This application allows users to upload up to 3 PDFs simultaneously, switch active documents using tab selectors, view automated structured summaries, query papers using an optimized smart chunking RAG system, generate APA/BibTeX citations, and study complex jargon using interactive 3D flippable flashcards.

Developed as a highly polished, production-ready technical assessment showcase.

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
Ensure you have Python installed, then run this command in your terminal to install the required libraries:
```bash
pip install flask groq pymupdf reportlab
```

### 2. Configure Your Groq API Key
1. Get a free high-speed API key instantly from the [Groq Console](https://console.groq.com/).
2. Open **`app.py`** in a text editor.
3. Locate line **33**:
   ```python
   GROQ_API_KEY = "YOUR_GROQ_API_KEY"
   ```
4. Replace `"YOUR_GROQ_API_KEY"` with your live key (e.g. `"gsk_XYZ..."`) and save the file.

### 3. Run the Application
1. Open your terminal or Command Prompt in the project folder:
   ```bash
   cd "C:\Users\aruns\OneDrive\Documents\coding\ai_research_assistant"
   ```
2. Launch the Flask server:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to:
   👉 **`http://localhost:5000`**

---

## 🌟 Advanced Features Implemented

### 1. Smart Lexical Chunking (Context Grounded RAG)
Dense academic papers can easily exceed LLM token limits and cause answer hallucination. 
* **The Solution**: If a paper exceeds 6,000 words, the backend divides it into overlapping sliding-window chunks (3,000 words each, 500-word overlap).
* **The Ranker**: When a user asks a question, a custom, pure-Python lexical overlap matcher filters out stopwords, calculates term frequencies inside each chunk, and **sends only the most relevant section** to Groq. 
* This secures grounded, citation-level accuracy, reduces latency under $1.5$ seconds, and displays an on-screen chunking notice to the user.

### 2. Interactive 3D Study Flashcards
* Pulls 6 crucial technical terms, concepts, or algorithms along with simple, intuitive study definitions.
* The frontend renders these cards in a responsive CSS Grid using hardware-accelerated **CSS 3D Transforms** (`perspective: 1000px`, `transform-style: preserve-3d`, and `backface-visibility: hidden`). Clicking any card physically flips it $180^\circ$ around the Y-axis to reveal the definition, making it an excellent active recall study tool.

### 3. Citation & BibTeX Generator
* Automatically extracts publication metadata (Title, Authors list, Journal, Volume, Issue, Pages, Publisher, Year) by scanning *only* the first two pages of the active PDF.
* Assembles and presents perfect copy-to-clipboard **APA Style** citations and valid **LaTeX BibTeX records** for bibliographies.

### 4. Professional UI/UX & Layout Polish
* Styled with a beautiful solid dark-mode palette (`#0f0f0f` background and royal violet `#8b5cf6` accents) with blur-backdrop glassmorphic modals.
* Fully constrained `100vh` grid layout prevents the window itself from overflowing—keeping the query bar anchored and forcing the chat timeline to scroll natively.

---

## 🛡️ Robust Engineering & Fail-Safes

* **Conversational LLM Filters**: AI models often return conversational text before/after their JSON responses (e.g. *"Sure! Here is the JSON: ..."*), which crashes standard JSON parsers. We solved this by using **Python Regular Expressions (`re.search`)** to surgically isolate *only* the structural JSON blocks (`[...]` or `{...}`) before passing them to `json.loads()`.
* **Casing Normalization**: Frontend JavaScript maps variable dictionary keys (e.g. `Term` vs `term`) to lowercase properties dynamically, avoiding blank card displays.
* **Audit Checks**: Uploaded files are verified for PDF extension compliance, audited to block files larger than 10MB, and checked for scanned image pages (triggering unreadable-PDF notifications).
* **Disk-Cached Text Mappings**: Scraped text is stored as companion `.txt` files under `/uploads` to prevent memory bottlenecks and session cookie overflow.

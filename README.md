# 📰 News Research Tool

A Streamlit-based web application that allows users to input URLs of news articles, process the content using NLP, and ask questions to extract meaningful information and insights from the articles. Powered by Hugging Face Transformers, FAISS vector search, and LangChain.

---

## 🚀 Features

- 🔗 Accepts up to 3 news article URLs
- 🧠 Uses **sentence-transformers** for semantic embeddings
- 📚 Splits article content into meaningful text chunks
- 🗃️ Stores and retrieves vector embeddings using **FAISS**
- 🤖 Answers user queries using `FLAN-T5` via LangChain’s RetrievalQA pipeline
- 💾 Saves and loads processed vectors from a local `.pkl` file

---

## 🛠️ Tech Stack

| Component     | Tool / Library                               |
|---------------|----------------------------------------------|
| Web App       | Streamlit                                    |
| LLM           | FLAN-T5 (via HuggingFace Transformers)       |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2       |
| Vector Store  | FAISS                                         |
| NLP Framework | LangChain                                    |
| Data Loading  | UnstructuredURLLoader                        |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/news-research-tool.git
cd news-research-tool
```

### 2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file
```env
# Optional, only if needed for API keys (e.g., HuggingFaceHub or OpenAI)
```

---

## ▶️ Usage

```bash
streamlit run app.py
```

1. Enter up to **3 article URLs** in the sidebar.
2. Click **"Process URLs"** to fetch and vectorize the content.
3. Ask questions about the articles in the main input box.
4. Get AI-generated answers sourced from the processed content.

---

## 📁 File Structure

```
news-research-tool/
├── app.py                  # Main Streamlit app
├── faiss_store_openai.pkl # Serialized vector store (auto-generated)
├── requirements.txt        # Required Python packages
├── README.md               # Project README
└── .env                    # (Optional) Environment variables
```

---

## 📌 Requirements

- Python 3.8+
- `streamlit`
- `langchain`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `unstructured`
- `beautifulsoup4`
- `python-dotenv`

You can install them with:

```bash
pip install streamlit langchain transformers sentence-transformers faiss-cpu unstructured beautifulsoup4 python-dotenv
```

---

## 🧠 Example Models Used

- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM (Q&A)**: `google/flan-t5-small`

---

## 🧪 Sample Questions You Can Try

- "What is the main issue discussed in the articles?"
- "Who are the key people involved?"
- "Summarize the event mentioned in all the URLs."

---

## 🙋‍♂️ Author

- 👨‍💻 [Your Name](https://github.com/your-username)
- 📫 Feel free to connect with me for questions or collaborations.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

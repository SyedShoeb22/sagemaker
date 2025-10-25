```markdown
# 🧠 Retrieval-Augmented Generation (RAG) on AWS SageMaker — No S3 Required

This guide shows how to implement a **Retrieval-Augmented Generation (RAG)** pipeline **entirely inside AWS SageMaker** — without using Amazon S3.  
All data and models run locally within SageMaker’s **EBS-backed storage**.

---

## 📘 Overview

**Retrieval-Augmented Generation (RAG)** =  
Retrieval (fetch relevant context) + Generation (answer using LLM)

We’ll build a minimal working RAG setup using:
- 🧩 `langchain` for orchestration  
- ⚙️ `sentence-transformers` for embeddings  
- 🧮 `faiss` for local vector search  
- 🤖 `transformers` for text generation  

All steps run inside **SageMaker Studio** or **SageMaker Notebook Instance**, using **local files only**.

---

## 🧩 Step 1: Environment Setup

Launch a new **SageMaker Studio** or **Notebook Instance**.

These environments have persistent storage under:
```

/home/ec2-user/SageMaker/

````

---

## 🧾 Step 2: Create a Local Dataset

In your SageMaker terminal:

```bash
mkdir -p ~/SageMaker/rag_demo/data
cd ~/SageMaker/rag_demo/data

# Create a small sample dataset file
cat > knowledge.txt <<'EOF'
Amazon SageMaker is a fully managed machine learning service provided by AWS.
It enables developers to build, train, and deploy machine learning models quickly.
SageMaker provides tools for data labeling, model monitoring, and MLOps pipelines.
EOF
````

---

## ⚙️ Step 3: Install Required Dependencies

In a SageMaker Jupyter notebook cell, install all libraries:

```python
!pip install langchain faiss-cpu sentence-transformers transformers
```

---

## 🧠 Step 4: Build the RAG Pipeline (Local Mode)

Create a new Jupyter notebook cell and paste the following code:

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# 1. Load local dataset
with open("/home/ec2-user/SageMaker/rag_demo/data/knowledge.txt", "r") as f:
    data = f.read()

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.create_documents([data])

# 3. Create embeddings and local FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# 4. Initialize a local LLM (Flan-T5)
llm_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=128
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# 5. Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# 6. Ask a question
query = "What is SageMaker used for?"
answer = qa.run(query)

print("Q:", query)
print("A:", answer)
```

---

## ✅ Step 5: Verify the Output

Example output:

```
Q: What is SageMaker used for?
A: It is used for building, training, and deploying machine learning models.
```

This confirms that the **RAG pipeline** is working end-to-end using only **local data and resources**.

---

## 💾 Step 6: Save and Reload FAISS Index (Optional)

You can persist your FAISS index locally:

```python
# Save index
vectorstore.save_local("/home/ec2-user/SageMaker/rag_demo/faiss_index")

# Later reload
from langchain.vectorstores import FAISS
new_vs = FAISS.load_local("/home/ec2-user/SageMaker/rag_demo/faiss_index", embedding_model)
```

## 📂 Folder Structure

```
~/SageMaker/rag_demo/
├── data/
│   └── knowledge.txt
├── faiss_index/
│   └── (auto-generated files)
└── rag_demo.ipynb
```

---

## 🧰 Requirements

* Python 3.8+
* SageMaker Notebook or Studio environment
* Internet access for model downloads

---

## 🚀 Summary

You’ve successfully:

* Built a **Retrieval-Augmented Generation (RAG)** pipeline
* Used **local files only** (no S3)
* Combined **FAISS**, **HuggingFace**, and **LangChain** inside SageMaker


---

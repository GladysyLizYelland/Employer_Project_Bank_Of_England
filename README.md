# 📌 Bank of England Employer Project: NLP Analysis of G-SIB Announcements

# 📖 Overview
This project is part of the University of Cambridge Career Accelerator, in collaboration with the Bank of England. The goal is to analyse quarterly earnings announcements from Global Systemically Important Banks (G-SIBs) using advanced Natural Language Processing (NLP) techniques.

By leveraging NLP methods such as topic modelling, sentiment analysis, and summarisation, we aim to provide insights that support financial stability assessments and risk monitoring.

*Note: This repository originated as a collaborative group project. I have since extended the codebase, expanded the scope, and integrated domain-specific architectures as an independent learning initiative.*

> [!NOTE]
> **Architectural Evolution & Model Upgrade:**
> *This repository originated as a collaborative group project. I have since extended the codebase, expanded the scope, and integrated domain-specific architectures as an independent learning initiative.*
> 
> As part of this expansion, I migrated the Tier-1 sentiment pipeline away from the generic base model (`ProsusAI/finbert`) to a specialized tonal variant (`yiyanghkust/finbert-tone`). While the original FinBERT baseline functions well on explicit financial news headlines, it suffers from a pronounced negativity bias when applied to long-form banking text—frequently misclassifying balanced risk-management procedures (such as adjusting loan loss provisions) as market drops. Upgrading to FinBERT-Tone allows the pipeline to accurately isolate objective corporate delivery, structural confidence metrics, and baseline policy shifts within the formal Prepared Remarks.
>
# 🎯 Objectives
Extract and process textual data from G-SIB earnings calls.
Identify key themes (topics) and emerging trends.
Conduct sentiment analysis.
Generate summary insights with instruction-tuned models.
Provide a structured dashboard or reporting format for key stakeholders.

# 📂 Project Structure

```text
.
├── bank_data/                   # Data storage layer
│   ├── raw/                     # Raw, unstructured transcripts and documents
│   ├── parsed/                  # Structured formats extracted from raw text
│   ├── preprocessed/            # Cleaned, tokenised text data optimized for NLP
│   ├── sentimented/             # Preprocessed data with sentiment scores/metrics added
│   └── topics/                  # Final processed text mapped with dominant themes/topics
│
├── notebooks/                   # R&D and analysis environment
│   ├── parser/                  # Raw text extraction and structured parsing pipelines
│   ├── eda/                     # Exploratory Data Analysis notebooks
│   ├── preprocessing/           # Text cleansing, tokenisation, and normalization routines
│   ├── sentiment_analysis/      # Sentiment scoring models and implementation i.e. Finbert
│   ├── topic_modeling/          # BERTopic pipelines optimised with FinTextSim vs. BGE baselines
│   ├── llm/                     # Prompt engineering and fine-tuning experiments (e.g., FinGPT, Gemini, Llama 3.3) 
│   ├── consolidate/             # Data merging, alignment, and schema aggregation
│   ├── summarisation.ipynb      # Text summarisation methods (extractive/abstractive)
│   └── visualization/           # Notebooks for generating analytical plots and charts
│
├── reports/                     # Generated insights, final documentation, and project findings
├── requirements.txt             # Python dependencies and library versions
├── .gitignore                   # Specified files and directories hidden from version control
└── README.md                    # Project documentation
```

## 🤖 Embedding Model Selection

This pipeline leverages sentence-transformer embeddings to vectorize text for downstream sentiment analysis and topic modeling. Before running the production scripts, you must configure the embedding model that best fits your target dataset.

While our initial team implementation utilised `BAAI/bge-base-en-v1.5`, I conducted further research and extended the architecture to support `JehnenS/FinTextSim`. Integrating this domain-specific financial model yields significantly better clustering outcomes by eliminating semantic noise inherent to general-purpose models.

> [!TIP]
> **Recommended Baseline:** Use `JehnenS/FinTextSim` for structural financial text analysis (e.g., corporate filings and earnings calls) and fallback to `BAAI/bge-base-en-v1.5` only if expanding the raw dataset to include general, non-financial news text.

### 🔍 Selection Logic

* **Use `JehnenS/FinTextSim` for `notebooks/topic_modeling/`**  
  This model is strictly optimized for corporate financial semantics. When paired with `BERTopic`, it isolates sharp, distinct economic clusters. Running standard models without financial tuning introduces high misclassification rates due to dense, specialized domain terminology.
  
* **Use `BAAI/bge-base-en-v1.5` for `notebooks/sentiment_analysis/` & Broad Retrieval**  
  If the `bank_data/raw/` directory expands to include broad macroeconomic news articles, customer letters, or generic texts, use the BGE base model to ensure robust out-of-domain coverage.

### ⚙️ How to Configure

Toggle the model initialization string inside your configuration layer or notebook initialization cells:

```python
# Configuration parameters for the execution pipeline

# 1. Preferred Choice: Specialized financial parsing & topic extraction
MODEL_PATH = "JehnenS/FinTextSim"

# 2. Alternative: Broad general-purpose semantic retrieval
# MODEL_PATH = "BAAI/bge-base-en-v1.5"
```


### 🚀  Our project roadmap

### ① Understand the problem and the data
 Bank of England problem  
↓  
 What information is contained in quarterly results?  
↓  
 Why might analyst–management conversations be informative?  
↓  
 Define what we want to investigate

**Output:** a clear analytical question and data strategy

---

### ② Acquire, structure and explore the transcripts

 Investor Relations websites  
↓  
 Earnings-call transcript PDFs  
↓  
 Document manifest  
↓  
 Page-level text  
↓  
 Complete transcript documents  
↓  
 Extraction and data-quality checks

**Output:** a clean, traceable corpus of quarterly-results transcripts

---

### ③ Turn transcripts into conversations -structured data - by parsing it

 Complete transcripts  
↓  
 Separate prepared remarks and Q&A  
↓  
 Identify speakers  
↓  
 Analyst vs management  
↓  
 Analyst question →  Management response

**Output:** structured speaker turns and Q&A pairs

---
###  ⑤ Preprocess the data to be used for Bertopic and Finbert - 

Find any duplicate data - and drop it and log
↓  
Find any missing answers or questions - and drop it and log
↓ 
clean text from any common greeting words, html encoding etc
↓ 
**Output:** structured Clean speaker turns and Q&A pairs text

---
### ⑥  Measure financial sentiment - thru finbert to

 Analyst questions +  Management responses  
↓  
 **FinBERT**  
↓  
 Positive |  Neutral |  Negative  
↓  
 Compare sentiment by speaker, topic and reporting period

**Question:** *Where do analysts appear concerned, and how does management respond?*

---

### ⑦  Discover the main themes - topics using Bertopib
**Notebook 5**

 Speaker turns / Q&A pairs  
↓  
 **Sentence Transformers** — semantic embeddings  
↓  
 **BERTopic** — topic discovery  
↓  
 Credit | Capital | Costs | Liquidity | Growth | ...

**Question:** *What are analysts and management actually talking about?*

---

###7 Extract and summarise insight with an LLM
**Notebook 6**

 Analyst question →  Management response  
↓  
 **such as Gemma 3, FinGPT etc**  
↓  
 Structured extraction

`Topic`  
`Analyst concern`  
`Management response`  
`Financial metrics mentioned`  
`Forward-looking statements`  
`Risk / uncertainty`

**Question:** *Can an LLM turn complex financial conversations into structured,
evidence-grounded information?*

---

### 8 Bring the signals together


In the final stage, we bring together the different signals extracted from the
earnings-call Q&A:

| Signal | What we have learned |
|---|---|
|  **Topics** | What analysts and management are talking about |
|  **Sentiment** | How positively, neutrally or negatively issues are discussed |
|  **Q&A behaviour** | What analysts ask and how management responds |
|  **LLM extraction** | Specific concerns, metrics, risks and uncertainty |

We then examine how these signals **change across reporting periods**.

**Topics + Sentiment + Q&A behaviour + LLM extraction**

↓

**Changes across reporting periods**

↓

###  Regulatory insight

We will investigate:

**What are analysts concerned about?**  
↓  
**How does management respond?**  
↓  
**Which topics are becoming more prominent or negative?**  
↓  
**What has changed relative to previous reporting periods?**  
↓  
**Do quarterly results announcements provide additional insight about the firm?**

---

###  Our main NLP toolkit

| Method | Model / approach | Why are we using it? |
|---|---|---|
| Text structuring | Python + regular expressions | Turn unstructured transcripts into analysable conversations |
| Financial sentiment | **FinBERT** | Understand sentiment in financial language |
| Semantic representation | **Sentence Transformers** | Represent passages according to meaning rather than individual words |
| Topic discovery | **BERTopic** | Discover recurring themes in analyst and management discussions |
| LLM extraction | **Gemma 3 4B Instruct** | Extract and summarise structured information from Q&A exchanges |

---

> ### The idea connecting the whole project
>
> We are not applying NLP models simply because they are available.
>
> Each stage helps us answer a progressively more useful question:
>
> **What was said? → Who said it? → What were they talking about? →  
> How was it expressed? → What concerns can we extract? → What changed? →  
> Does this provide useful additional insight about the firm?**



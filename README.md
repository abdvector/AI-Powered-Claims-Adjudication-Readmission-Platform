# Smart-Care: AI-Powered Healthcare Document Intelligence & Claims Adjudication Platform

A cloud-native clinical intelligence platform designed to automate hospital discharge parsing, predict 30-day patient readmission risk using Explainable AI, and streamline healthcare insurance claims adjudication.

---

## Key Features

1. **Multimodal Clinical Document Parsing & OCR**
   - Extracts structured clinical entities from digital PDFs, DOCX, and scanned hospital invoices.
   - Dual-engine fallback: lightweight `pypdf` for digital charts and Google Gemini 2.5 Flash Vision for noisy/handwritten clinical scans.
   - Weighted OCR confidence scoring algorithm with medical token prioritization.

2. **30-Day Hospital Readmission Risk Modeling (ML & Econometrics)**
   - Calibrated statistical engine integrating Charlson Comorbidity Index, Inpatient Length of Stay (LOS), Polypharmacy, and Social Determinants of Health (SDoH).
   - Patient-level Explainable AI (XAI) feature attribution (log-odds impact breakdown).
   - Econometric cost-minimization theorem analysis comparing readmission penalties against targeted care-management interventions.

3. **Claims Adjudication Queue & Side-by-Side In-App Document Review**
   - Adjudication queue backed by Supabase PostgreSQL for live claim status auditing.
   - Interactive in-app visual PDF inspector alongside extracted clinical records and risk metrics.
   - One-click adjudication decisions: *Approve Claim*, *Flag for Audit*, and *Deny Preventable Readmission*.

4. **Evidence-Grounded RAG Clinical Search**
   - High-dimensional 768-vector embeddings via Gemini `text-embedding-004`.
   - Cosine similarity vector search backed by Supabase `pgvector` RPC functions.
   - Interactive conversational clinical assistant providing grounded citations for auditability.

---

## Technology Stack

* **Frontend & Dashboard:** Streamlit, Plotly Express & Graph Objects
* **Multimodal OCR & LLM:** Google Gemini 2.5 Flash (`google-genai` SDK)
* **Embeddings & Vector Search:** Gemini `text-embedding-004`, Supabase `pgvector`
* **Database & Cloud Storage:** Supabase PostgreSQL & Supabase Object Storage
* **Machine Learning:** XGBoost, Scikit-Learn, NumPy, Pandas

---

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file with your free-tier cloud credentials:
```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Launch Application
```bash
streamlit run app.py
```

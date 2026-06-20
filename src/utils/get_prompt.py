"""
Prompt Generation Service for Clinical Entity Extraction and RAG
"""
from typing import List, Dict, Any

def get_metadata_extraction_prompt(text: str, user_id: str = "default_global") -> str:
    return f"""
You are an expert healthcare and insurance document understanding system.

Analyze the medical/insurance document text and extract structured clinical and demographic metadata.

Return ONLY valid JSON with no markdown backticks or explanations.

Schema:
{{
    "document_type": "",
    "document_title": "",
    "document_number": "",
    "patient_id": "",
    "patient_name": "",
    "dob": null,
    "gender": "",
    "entity_name": "",
    "document_date": null,
    "admission_date": null,
    "discharge_date": null,
    "length_of_stay": 0,
    "primary_diagnosis": "",
    "comorbidities": [],
    "num_medications": 0,
    "follow_up_scheduled": false,
    "lives_alone": false,
    "metadata": {{}}
}}

Field Definitions & Rules:
- document_type: MUST be one of: ["discharge summary", "clinical record", "medical claim", "aadhaar", "invoice", "other"].
- document_title: Main heading (e.g. "Hospital Discharge Summary", "Inpatient Progress Note").
- document_number: Document/Claim reference number or bill invoice ID.
- patient_id: Unique patient identifier (MRN, Aadhaar number, or Patient ID). If none found, generate an identifier like "PAT-AUTO".
- patient_name / entity_name: Full name of the patient.
- dob: Date of birth (YYYY-MM-DD format). If unavailable, return null.
- gender: "Male", "Female", or "Other".
- document_date / discharge_date: The date of discharge or document creation (YYYY-MM-DD).
- admission_date: The date admitted to the hospital (YYYY-MM-DD).
- length_of_stay: Integer number of inpatient days (e.g., 4). If difference between admission and discharge is clear, calculate it.
- primary_diagnosis: The main clinical condition treated (e.g., "Acute Congestive Heart Failure", "Type 2 Diabetes with Ketoacidosis", "COPD Exacerbation").
- comorbidities: Array of all secondary chronic conditions mentioned (e.g., ["Hypertension", "Diabetes", "Chronic Kidney Disease", "Asthma"]).
- num_medications: Total count of discharge medications prescribed.
- follow_up_scheduled: Boolean (true if a 7-day or post-discharge outpatient clinic visit is explicitly scheduled; false otherwise).
- lives_alone: Boolean (true if doctor noted patient lives alone, lacks home caregiver support, or lives in isolated housing; false otherwise).
- metadata: Extract any additional key-value clinical or financial observations (e.g. attending physician, billed amount, vital signs, discharge instructions).

DOCUMENT TEXT:
{text}
"""

def get_rag_response_prompt(user_query: str, retrieved_contexts: List[Dict[str, Any]]) -> str:
    """Generates prompt for RAG Clinical Audit Assistant."""
    context_str = ""
    for i, doc in enumerate(retrieved_contexts, 1):
        p_id = doc.get("patient_id", "N/A")
        diag = doc.get("primary_diagnosis", "N/A")
        risk = doc.get("readmission_risk", "N/A")
        text = doc.get("extracted_text", "")[:1200]
        context_str += f"\n--- DOCUMENT {i} (Patient: {p_id} | Diagnosis: {diag} | Readmission Risk: {risk}) ---\n{text}\n"

    return f"""
You are a Clinical & Insurance Claims Audit AI Assistant.

Your task is to answer the Claims Adjuster's question using ONLY the retrieved clinical context provided below.

USER QUERY:
{user_query}

RETRIEVED CLINICAL CONTEXT:
{context_str}

INSTRUCTIONS:
1. Provide a direct, professional, and evidence-grounded answer.
2. If the question asks about readmission risk, care compliance, or clinical diagnoses, cite the specific patient ID and document facts.
3. If the retrieved documents do not contain enough information to answer the question, clearly state that the current records do not specify it.
4. Keep the answer concise, structured with bullet points where appropriate, and actionable for insurance auditing.
"""

def get_search_summary_prompt(user_query: str, search_type: str, search_results: list) -> str:
    return f"""
You are an intelligent healthcare search assistant.
 
Summarize how the returned medical records match the adjuster's query:

SEARCH QUERY: {user_query}
SEARCH TYPE: {search_type}
RESULTS: {search_results}

Synthesize findings into a concise, professional paragraph explaining which patient records best match and highlighting any high-risk readmissions.
"""
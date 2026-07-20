"""
Supabase Client & Database Helper Service
Handles all operations for patients, clinical records, deduplication hashes, and vector semantic search.
"""
import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from google import genai
from src.config.config import SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY

# Initialize Supabase Client
_supabase: Optional[Client] = None

def get_supabase() -> Optional[Client]:
    """Returns the singleton Supabase client instance."""
    global _supabase
    if _supabase is None:
        url = SUPABASE_URL or os.getenv("SUPABASE_URL")
        key = SUPABASE_KEY or os.getenv("SUPABASE_KEY")
        if url and key:
            try:
                _supabase = create_client(url, key)
            except Exception as e:
                print(f"[-] Failed to initialize Supabase client: {e}")
                return None
    return _supabase

# Initialize Gemini Client for Embeddings
_genai_client = None
def get_genai_client():
    global _genai_client
    if _genai_client is None:
        key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if key:
            _genai_client = genai.Client(api_key=key)
    return _genai_client

def generate_embedding(text: str) -> Optional[List[float]]:
    """Generates 768-dimensional text embedding via Gemini free embedding model with fallback."""
    if not text or not text.strip():
        return None
    try:
        client = get_genai_client()
        if not client:
            return None
        cleaned_text = text.strip()[:6000]
        
        for m in ["text-embedding-004", "embedding-001"]:
            try:
                response = client.models.embed_content(
                    model=m,
                    contents=cleaned_text
                )
                if response and response.embeddings:
                    return response.embeddings[0].values
            except Exception:
                continue
    except Exception as e:
        print(f"[-] Embedding generation failed: {e}")
    return None

# ==========================================
# PATIENT OPERATIONS
# ==========================================
def upsert_patient(patient_id: str, name: str = "", dob: str = None, gender: str = "") -> bool:
    """Upserts patient record in 'patients' table."""
    sb = get_supabase()
    if not sb or not patient_id:
        return False
    try:
        data = {
            "patient_id": str(patient_id).strip(),
            "name": name or "Unknown Patient",
            "dob": dob if dob and dob != "null" else None,
            "gender": gender or "Unknown"
        }
        sb.table("patients").upsert(data).execute()
        return True
    except Exception as e:
        print(f"[-] Failed to upsert patient: {e}")
        return False

# ==========================================
# CLINICAL RECORD OPERATIONS
# ==========================================
def insert_clinical_record(record: Dict[str, Any]) -> Optional[str]:
    """
    Inserts a parsed clinical record into 'clinical_records' table.
    Returns the created record_id UUID if successful.
    """
    sb = get_supabase()
    if not sb:
        return None
    try:
        # 1. Ensure patient exists
        patient_id = record.get("patient_id")
        if patient_id:
            upsert_patient(
                patient_id=patient_id,
                name=record.get("entity_name") or record.get("patient_name"),
                dob=record.get("dob"),
                gender=record.get("gender")
            )

        # 2. Generate embedding for RAG / Semantic Search if text provided
        raw_text = record.get("extracted_text", "")
        embedding = None
        if raw_text:
            embedding = generate_embedding(raw_text)

        payload = {
            "patient_id": patient_id,
            "file_name": record.get("file_name", "unknown_document.pdf"),
            "document_number": record.get("document_number"),
            "entity_name": record.get("entity_name") or record.get("patient_name"),
            "admission_date": record.get("admission_date"),
            "discharge_date": record.get("discharge_date"),
            "length_of_stay": int(record.get("length_of_stay", 0) or 0),
            "primary_diagnosis": record.get("primary_diagnosis"),
            "comorbidities": record.get("comorbidities", []),
            "num_medications": int(record.get("num_medications", 0) or 0),
            "follow_up_scheduled": bool(record.get("follow_up_scheduled", False)),
            "lives_alone": bool(record.get("lives_alone", False)),
            "readmission_risk": float(record.get("readmission_risk", 0.0) or 0.0),
            "adjudication_status": record.get("adjudication_status", "Pending Review"),
            "review_reason": record.get("review_reason"),
            "extracted_text": raw_text,
            "embedding": embedding
        }

        # Filter out keys with None dates if invalid
        for date_key in ["admission_date", "discharge_date"]:
            if payload[date_key] in ["null", "None", "", None]:
                payload[date_key] = None

        res = sb.table("clinical_records").insert(payload).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get("record_id")
    except Exception as e:
        print(f"[-] Failed to insert clinical record: {e}")
    return None

def update_clinical_record(record_id: str, updates: Dict[str, Any]) -> bool:
    """Updates specific fields of a clinical record."""
    sb = get_supabase()
    if not sb or not record_id:
        return False
    try:
        sb.table("clinical_records").update(updates).eq("record_id", record_id).execute()
        return True
    except Exception as e:
        print(f"[-] Failed to update clinical record: {e}")
        return False

def get_clinical_records(status_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches clinical records ordered by created_at DESC."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        query = sb.table("clinical_records").select("*, patients(name, dob, gender)").order("created_at", desc=True).limit(limit)
        if status_filter and status_filter != "All":
            query = query.eq("adjudication_status", status_filter)
        res = query.execute()
        return res.data or []
    except Exception as e:
        print(f"[-] Failed to fetch clinical records: {e}")
        return []

def get_clinical_record_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single clinical record by record_id."""
    sb = get_supabase()
    if not sb or not record_id:
        return None
    try:
        res = sb.table("clinical_records").select("*, patients(*)").eq("record_id", record_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
    except Exception as e:
        print(f"[-] Failed to fetch record by id: {e}")
    return None

# ==========================================
# VECTOR & KEYWORD SEARCH (RAG)
# ==========================================
def semantic_search_records(query_text: str, match_threshold: float = 0.25, match_count: int = 5) -> List[Dict[str, Any]]:
    """Performs Cosine Similarity vector search via Supabase RPC function."""
    sb = get_supabase()
    if not sb or not query_text:
        return []
    try:
        query_emb = generate_embedding(query_text)
        if not query_emb:
            return []
        
        # Call Supabase RPC function 'match_clinical_records'
        params = {
            "query_embedding": query_emb,
            "match_threshold": match_threshold,
            "match_count": match_count
        }
        res = sb.rpc("match_clinical_records", params).execute()
        return res.data or []
    except Exception as e:
        print(f"[-] Semantic vector search failed: {e}")
        return []

def keyword_search_records(query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Performs keyword text search over clinical records."""
    sb = get_supabase()
    if not sb or not query_text:
        return []
    try:
        term = f"%{query_text.strip()}%"
        res = sb.table("clinical_records").select("record_id, patient_id, file_name, primary_diagnosis, readmission_risk, adjudication_status, extracted_text")\
            .or_(f"primary_diagnosis.ilike.{term},entity_name.ilike.{term},file_name.ilike.{term},extracted_text.ilike.{term}")\
            .limit(limit).execute()
        return res.data or []
    except Exception as e:
        print(f"[-] Keyword search failed: {e}")
        return []

# ==========================================
# DEDUPLICATION HASH OPERATIONS
# ==========================================
def check_exact_duplicate(sha256_hash: str) -> bool:
    """Checks if exact SHA-256 hash exists in 'document_hashes' table."""
    sb = get_supabase()
    if not sb or not sha256_hash:
        return False
    try:
        res = sb.table("document_hashes").select("sha256_hash").eq("sha256_hash", sha256_hash).execute()
        return len(res.data or []) > 0
    except Exception as e:
        print(f"[-] Exact duplicate check failed: {e}")
        return False

def get_all_document_hashes() -> List[Dict[str, Any]]:
    """Returns all MinHash and pHash signatures for near-duplicate math."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        res = sb.table("document_hashes").select("sha256_hash, minhash_signature, phash_signature").execute()
        return res.data or []
    except Exception as e:
        print(f"[-] Failed to fetch document hashes: {e}")
        return []

def insert_document_hash(sha256_hash: str, record_id: Optional[str], minhash_sig: str = "", phash_sig: str = "") -> bool:
    """Saves document signatures into 'document_hashes' table."""
    sb = get_supabase()
    if not sb or not sha256_hash:
        return False
    try:
        data = {
            "sha256_hash": sha256_hash,
            "record_id": record_id,
            "minhash_signature": minhash_sig or "",
            "phash_signature": phash_sig or ""
        }
        sb.table("document_hashes").upsert(data).execute()
        return True
    except Exception as e:
        print(f"[-] Failed to insert document hash: {e}")
        return False

def check_data_level_duplicate(doc_num: str, entity_name: str) -> bool:
    """Checks if a clinical record with matching document number & patient name already exists."""
    sb = get_supabase()
    if not sb or not doc_num or not entity_name:
        return False
    try:
        res = sb.table("clinical_records").select("record_id")\
            .eq("document_number", str(doc_num).strip())\
            .eq("entity_name", str(entity_name).strip())\
            .execute()
        return len(res.data or []) > 0
    except Exception as e:
        print(f"[-] Data-level duplicate check failed: {e}")
        return False

# ==========================================
# SUPABASE FILE STORAGE (Free Tier Object Storage)
# ==========================================
STORAGE_BUCKET_NAME = "clinical-docs"

def ensure_storage_bucket():
    """Ensures the clinical-documents bucket exists in Supabase Storage."""
    sb = get_supabase()
    if not sb:
        return False
    try:
        buckets = sb.storage.list_buckets()
        bucket_names = [b.name for b in buckets] if buckets else []
        if STORAGE_BUCKET_NAME not in bucket_names:
            try:
                sb.storage.create_bucket(STORAGE_BUCKET_NAME, options={"public": True})
            except Exception:
                pass
        return True
    except Exception as e:
        print(f"[-] Bucket check error: {e}")
        return False

def upload_document_to_storage(file_bytes: bytes, file_name: str) -> Optional[str]:
    """
    Uploads a document to Supabase Storage Bucket 'clinical-documents' (Free Tier).
    Returns the public/accessible URL of the uploaded document.
    """
    sb = get_supabase()
    if not sb or not file_bytes or not file_name:
        return None
    try:
        ensure_storage_bucket()
        import mimetypes
        mime, _ = mimetypes.guess_type(file_name)
        content_type = mime or ("application/pdf" if file_name.lower().endswith(".pdf") else "application/octet-stream")
        
        # Clean path
        clean_name = file_name.replace(" ", "_")
        
        # Upload with upsert
        sb.storage.from_(STORAGE_BUCKET_NAME).upload(
            path=clean_name,
            file=file_bytes,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        # Return public URL
        url_res = sb.storage.from_(STORAGE_BUCKET_NAME).get_public_url(clean_name)
        return url_res
    except Exception as e:
        print(f"[-] Failed to upload to Supabase Storage: {e}")
        return None

def download_document_from_storage(file_name: str) -> Optional[bytes]:
    """Downloads raw file bytes from Supabase Storage for in-app PDF rendering."""
    sb = get_supabase()
    if not sb or not file_name:
        return None
    try:
        clean_name = file_name.replace(" ", "_")
        data = sb.storage.from_(STORAGE_BUCKET_NAME).download(clean_name)
        return data
    except Exception as e:
        print(f"[-] Failed to download from Supabase Storage: {e}")
        return None


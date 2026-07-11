"""
Streamlit View: 1_Search.py
Dual-Tab Clinical Search & RAG AI Audit Assistant backed by Supabase pgvector & Gemini.
"""
import streamlit as st
import pandas as pd
from google import genai
from src.config.config import GEMINI_API_KEY
from src.utils.supabase_client import (
    semantic_search_records,
    keyword_search_records,
    get_clinical_records,
    download_document_from_storage
)
from src.utils.get_prompt import get_rag_response_prompt

st.markdown("""
<div style="
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    margin-bottom: 1.5rem;
">
    <h1 style="margin: 0; font-family: 'Outfit', sans-serif; color: #0F172A; font-size: 2.2rem; font-weight: 700;">Clinical Document Search & RAG Assistant</h1>
    <p style="margin: 6px 0 0 0; color: #475569; font-size: 1.05rem;">Search patient discharge records via vector similarity and converse with clinical charts via RAG.</p>
</div>
""", unsafe_allow_html=True)

tab_registry, tab_rag_chat = st.tabs([
    "Document Registry Search",
    "RAG Clinical Audit Assistant"
])

# =========================================================================
# TAB 1: DOCUMENT REGISTRY SEARCH
# =========================================================================
with tab_registry:
    st.markdown("### Search Medical & Claims Registry")
    
    c_search, c_mode = st.columns([3.5, 1.5])
    with c_search:
        search_query = st.text_input("Enter clinical condition, patient name, or keywords...", placeholder="e.g. Heart failure patients with no follow-up", label_visibility="collapsed")
    with c_mode:
        search_mode = st.selectbox("Search Mode", ["Semantic Vector Search (AI)", "Keyword Match"], label_visibility="collapsed")

    col_btn, col_filter = st.columns([1.5, 3.5])
    with col_btn:
        exec_search = st.button("Search Records", type="primary", use_container_width=True)
    with col_filter:
        risk_filter = st.selectbox("Filter by Readmission Risk", ["All Risk Tiers", "High Risk (>=60%)", "Moderate Risk (30-60%)", "Low Risk (<30%)"], index=0)

    results = []
    if exec_search or search_query:
        with st.spinner("Querying Supabase database..."):
            if "Semantic" in search_mode:
                results = semantic_search_records(search_query, match_threshold=0.15, match_count=15)
            else:
                results = keyword_search_records(search_query, limit=15)
    else:
        results = get_clinical_records(limit=15)

    if results and risk_filter != "All Risk Tiers":
        if "High" in risk_filter:
            results = [r for r in results if float(r.get("readmission_risk", 0.0) or 0.0) >= 0.60]
        elif "Moderate" in risk_filter:
            results = [r for r in results if 0.30 <= float(r.get("readmission_risk", 0.0) or 0.0) < 0.60]
        elif "Low" in risk_filter:
            results = [r for r in results if float(r.get("readmission_risk", 0.0) or 0.0) < 0.30]

    st.markdown(f"**Found {len(results)} matching records**")

    if results:
        for idx, doc in enumerate(results):
            p_id = doc.get("patient_id", "N/A")
            fname = doc.get("file_name", "Document")
            diag = doc.get("primary_diagnosis", "Not Specified")
            risk = float(doc.get("readmission_risk", 0.0) or 0.0) * 100
            status = doc.get("adjudication_status", "Pending Review")
            sim = doc.get("similarity")

            badge_color = "#DC2626" if risk >= 60 else ("#D97706" if risk >= 30 else "#16A34A")
            status_color = "#2563EB" if status == "Approved" else ("#DC2626" if "Audit" in status else "#64748B")

            similarity_badge = f"<span style='background:#F1F5F9; color:#334155; padding:3px 8px; border-radius:4px; font-size:12px; border:1px solid #CBD5E1; margin-right:8px;'>Match: {sim*100:.1f}%</span>" if sim is not None else ""

            with st.expander(f"Record: {p_id} — {diag} | {fname}"):
                c1, c2, c3 = st.columns([2, 2, 2])
                with c1:
                    st.markdown(f"**Patient ID:** `{p_id}`")
                    st.markdown(f"**Primary Diagnosis:** {diag}")
                with c2:
                    st.markdown(f"**Readmission Risk:** <span style='color:{badge_color}; font-weight:700; font-size:15px;'>{risk:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:600;'>{status}</span>", unsafe_allow_html=True)
                with c3:
                    if similarity_badge:
                        st.markdown(similarity_badge, unsafe_allow_html=True)
                    st.markdown(f"**File:** `{fname}`")

                st.markdown("---")
                doc_col_view, doc_col_text = st.columns([1, 1])
                with doc_col_view:
                    st.markdown("##### Visual Document Viewer")
                    from streamlit_pdf_viewer import pdf_viewer
                    
                    doc_bytes = st.session_state.get(f"doc_bytes_{fname}")
                    if not doc_bytes:
                        doc_bytes = download_document_from_storage(fname)
                        if doc_bytes:
                            st.session_state[f"doc_bytes_{fname}"] = doc_bytes

                    if doc_bytes:
                        if fname.lower().endswith(".pdf"):
                            try:
                                pdf_viewer(input=doc_bytes, width=420, height=400)
                            except Exception:
                                st.caption("Visual viewer unavailable.")
                        elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
                            st.image(doc_bytes, use_container_width=True)
                        else:
                            st.caption("Binary format preview not supported.")
                    else:
                        st.caption("Visual file binary not available in storage.")

                with doc_col_text:
                    st.markdown("##### Extracted Clinical Text")
                    st.text_area("Record Text", value=doc.get("extracted_text", "No text stored.")[:1500], height=400, key=f"text_preview_{idx}", disabled=True)
    else:
        st.info("No matching clinical records found. Try adjusting your query or upload new discharge summaries in the Upload tab.")

# =========================================================================
# TAB 2: RAG CLINICAL AUDIT ASSISTANT
# =========================================================================
with tab_rag_chat:
    st.markdown("### Conversational Record Analysis (RAG)")
    st.caption("Ask questions across all indexed clinical records. Answers are strictly grounded in retrieved evidence.")

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your RAG Clinical Audit Assistant. You can ask me questions about patient discharge plans, high-risk readmission patterns, or specific medication reconciliations."
            }
        ]

    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask a clinical or claims question (e.g. 'Which heart failure patients live alone?')"):
        st.session_state.rag_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving clinical records and synthesizing answer..."):
                retrieved_docs = semantic_search_records(user_prompt, match_threshold=0.20, match_count=4)
                
                if not retrieved_docs:
                    retrieved_docs = keyword_search_records(user_prompt, limit=3)

                if retrieved_docs:
                    rag_prompt = get_rag_response_prompt(user_prompt, retrieved_docs)

                    try:
                        client = genai.Client(api_key=GEMINI_API_KEY)
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=rag_prompt
                        )
                        answer_text = response.text if response and response.text else "Unable to generate answer."
                    except Exception as e:
                        answer_text = f"Error communicating with Gemini: {e}"

                    st.markdown(answer_text)

                    st.markdown("---")
                    st.markdown("##### Source Citations (Evidence Grounding):")
                    for i, r in enumerate(retrieved_docs, 1):
                        p_id = r.get("patient_id", "N/A")
                        diag = r.get("primary_diagnosis", "N/A")
                        fname = r.get("file_name", "Document")
                        risk = float(r.get("readmission_risk", 0.0) or 0.0) * 100
                        st.markdown(f"- **[{i}] Patient `{p_id}`** ({diag}) | Readmission Risk: **{risk:.1f}%** | `{fname}`")

                    st.session_state.rag_messages.append({"role": "assistant", "content": answer_text})
                else:
                    fallback_msg = "I searched your Supabase clinical records, but could not find relevant documentation to answer your question. Please ensure clinical documents have been uploaded."
                    st.markdown(fallback_msg)
                    st.session_state.rag_messages.append({"role": "assistant", "content": fallback_msg})

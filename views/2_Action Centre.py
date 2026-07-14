"""
Streamlit View: 2_Action Centre.py
Claims Adjudication Queue with Side-by-Side In-App Document Viewer & Audit Controls.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_pdf_viewer import pdf_viewer
from src.utils.supabase_client import (
    get_clinical_records,
    get_clinical_record_by_id,
    update_clinical_record,
    download_document_from_storage
)

st.markdown("""
<div style="
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    margin-bottom: 1.5rem;
">
    <h1 style="margin: 0; font-family: 'Outfit', sans-serif; color: #0F172A; font-size: 2.2rem; font-weight: 700;">Claims Adjudication & Audit Queue</h1>
    <p style="margin: 6px 0 0 0; color: #475569; font-size: 1.05rem;">Review, audit, and approve hospital discharge claims with side-by-side visual document inspection.</p>
</div>
""", unsafe_allow_html=True)

# State management for Side-by-Side Review Mode
if "review_record_id" not in st.session_state:
    st.session_state.review_record_id = None

# =========================================================================
# 1. SIDE-BY-SIDE IN-APP DOCUMENT REVIEW MODE
# =========================================================================
if st.session_state.review_record_id is not None:
    claim = get_clinical_record_by_id(st.session_state.review_record_id)
    
    if not claim:
        st.error("Claim record not found.")
        if st.button("Return to Queue"):
            st.session_state.review_record_id = None
            st.rerun()
    else:
        # Header with back navigation
        col_hdr, col_back = st.columns([4, 1])
        with col_hdr:
            st.markdown(f"### Reviewing Claim: `{claim.get('patient_id', 'N/A')}` — {claim.get('file_name', 'Document')}")
        with col_back:
            if st.button("Back to Queue", use_container_width=True):
                st.session_state.review_record_id = None
                st.rerun()
                
        st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0; border: none; border-bottom: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        col_doc, col_data = st.columns([1.1, 0.9], gap="large")

        # --- LEFT COLUMN: VISUAL DOCUMENT VIEWER ---
        with col_doc:
            st.markdown("#### Document Visual Inspection")
            file_name = claim.get("file_name", "")
            ext = file_name.split(".")[-1].lower() if "." in file_name else "pdf"

            # Retrieve file bytes from session cache or Supabase Storage
            cached_bytes = st.session_state.get(f"doc_bytes_{file_name}")
            if not cached_bytes:
                with st.spinner("Fetching document from cloud storage..."):
                    cached_bytes = download_document_from_storage(file_name)
                    if cached_bytes:
                        st.session_state[f"doc_bytes_{file_name}"] = cached_bytes

            if cached_bytes:
                if ext == "pdf":
                    try:
                        pdf_viewer(input=cached_bytes, width=650, height=720)
                    except Exception as e:
                        st.warning("Visual PDF viewer unavailable. Showing raw text below.")
                elif ext in ["jpg", "jpeg", "png"]:
                    st.image(cached_bytes, use_container_width=True)
                else:
                    st.info(f"Document format ({ext}) displayed via extracted text.")
            else:
                st.info("Document binary not in local cache; displaying extracted OCR text preview below.")

            with st.expander("Extracted OCR Text Content", expanded=False):
                st.text_area("Full Document Text", value=claim.get("extracted_text", "No text recorded."), height=250, disabled=True)

        # --- RIGHT COLUMN: CLINICAL DATA & ADJUDICATION ---
        with col_data:
            st.markdown("#### Clinical Analysis & Adjudication")
            
            risk = float(claim.get("readmission_risk", 0.0) or 0.0) * 100
            risk_color = "#DC2626" if risk >= 60 else ("#D97706" if risk >= 30 else "#16A34A")
            status = claim.get("adjudication_status", "Pending Review")
            
            st.markdown(f"""
            <div style='background:#F8FAFC; border:1px solid #E2E8F0; padding:16px; border-radius:8px; margin-bottom:15px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:14px; font-weight:600; color:#64748B;'>30-Day Readmission Risk</span>
                    <span style='color:{risk_color}; font-size:20px; font-weight:700;'>{risk:.1f}%</span>
                </div>
                <div style='margin-top:6px; font-size:14px;'>
                    <b>Current Status:</b> <code>{status}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Patient & Encounter Profile")
            p_data = claim.get("patients") or {}
            
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                st.markdown(f"**Patient Name:** {claim.get('entity_name') or p_data.get('name', 'N/A')}")
                st.markdown(f"**Admission:** {claim.get('admission_date', 'N/A')}")
                st.markdown(f"**Length of Stay:** {claim.get('length_of_stay', 0)} days")
            with p_c2:
                st.markdown(f"**Patient ID:** `{claim.get('patient_id', 'N/A')}`")
                st.markdown(f"**Discharge:** {claim.get('discharge_date', 'N/A')}")
                st.markdown(f"**Discharge Meds:** {claim.get('num_medications', 0)} prescribed")

            st.markdown("---")
            st.markdown("##### Clinical Diagnoses & SDoH Risk Factors")
            st.markdown(f"**Primary Diagnosis:** {claim.get('primary_diagnosis', 'N/A')}")
            comorbs = claim.get("comorbidities") or []
            st.markdown(f"**Chronic Comorbidities:** {', '.join(comorbs) if comorbs else 'None'}")
            
            follow_up = claim.get("follow_up_scheduled", False)
            lives_alone = claim.get("lives_alone", False)
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown(f"**7-Day Follow-Up:** {'Yes (Compliant)' if follow_up else 'No (Risk Factor)'}")
            with f_col2:
                st.markdown(f"**Lives Alone (SDoH):** {'Yes (Vulnerable)' if lives_alone else 'No'}")

            st.markdown("---")
            st.markdown("##### Adjudication Decision")
            
            rec_id = claim.get("record_id")
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("Approve Claim", key=f"side_app_{rec_id}", type="primary", use_container_width=True):
                    update_clinical_record(rec_id, {"adjudication_status": "Approved", "review_reason": "Approved in Side-by-Side Review"})
                    st.success("Claim Approved.")
                    st.session_state.review_record_id = None
                    st.rerun()
            with b2:
                if st.button("Flag for Audit", key=f"side_flag_{rec_id}", use_container_width=True):
                    update_clinical_record(rec_id, {"adjudication_status": "Flagged for Audit", "review_reason": "Flagged during Visual Clinical Inspection"})
                    st.warning("Claim Flagged.")
                    st.session_state.review_record_id = None
                    st.rerun()
            with b3:
                if st.button("Deny Payout", key=f"side_deny_{rec_id}", use_container_width=True):
                    update_clinical_record(rec_id, {"adjudication_status": "Denied", "review_reason": "Denied: Preventable 30-Day Readmission Protocol Deficiency"})
                    st.error("Claim Denied.")
                    st.session_state.review_record_id = None
                    st.rerun()

# =========================================================================
# 2. MAIN CLAIMS QUEUE TABLE VIEW
# =========================================================================
else:
    records = get_clinical_records(limit=100)
    df = pd.DataFrame(records) if records else pd.DataFrame()

    total_claims = len(df)
    pending_count = len(df[df["adjudication_status"] == "Pending Review"]) if not df.empty and "adjudication_status" in df else 0
    flagged_count = len(df[df["adjudication_status"] == "Flagged for Audit"]) if not df.empty and "adjudication_status" in df else 0
    approved_count = len(df[df["adjudication_status"] == "Approved"]) if not df.empty and "adjudication_status" in df else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Claims", total_claims)
    with m2:
        st.metric("Pending Review", pending_count)
    with m3:
        st.metric("Flagged for Audit", flagged_count, delta="Action Required" if flagged_count > 0 else None, delta_color="inverse")
    with m4:
        st.metric("Approved Payouts", approved_count)

    st.markdown("---")

    filter_col1, filter_col2 = st.columns([2, 3])
    with filter_col1:
        status_select = st.selectbox("Filter by Adjudication Status", ["All", "Pending Review", "Flagged for Audit", "Approved", "Denied"], index=0)

    filtered_records = get_clinical_records(status_filter=status_select, limit=50)

    st.markdown(f"### Claims Queue ({len(filtered_records)} Records)")

    if not filtered_records:
        st.info("No claims matching the selected filter in the Supabase database.")
    else:
        for idx, claim_row in enumerate(filtered_records):
            rec_id = claim_row.get("record_id")
            p_id = claim_row.get("patient_id", "N/A")
            fname = claim_row.get("file_name", "Discharge Note")
            diag = claim_row.get("primary_diagnosis", "Clinical Diagnosis")
            risk = float(claim_row.get("readmission_risk", 0.0) or 0.0) * 100
            status_val = claim_row.get("adjudication_status", "Pending Review")

            risk_color = "#DC2626" if risk >= 60 else ("#D97706" if risk >= 30 else "#16A34A")

            with st.container(border=True):
                c_row1, c_row2, c_row3, c_row4 = st.columns([2.2, 2.5, 1.8, 1.8])
                with c_row1:
                    st.markdown(f"**Patient ID:** `{p_id}`")
                    st.caption(f"File: {fname}")
                with c_row2:
                    st.markdown(f"**Diagnosis:** {diag}")
                    st.caption(f"Admit: {claim_row.get('admission_date', 'N/A')} | Discharge: {claim_row.get('discharge_date', 'N/A')}")
                with c_row3:
                    st.markdown(f"**Risk:** <span style='color:{risk_color}; font-weight:700;'>{risk:.1f}%</span>", unsafe_allow_html=True)
                    st.caption(f"Status: {status_val}")
                with c_row4:
                    if st.button("Review Document", key=f"btn_rev_{rec_id}_{idx}", use_container_width=True):
                        st.session_state.review_record_id = rec_id
                        st.rerun()

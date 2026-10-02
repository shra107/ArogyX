"""
app.py  —  ArogyX Healthcare Billing Fraud Detector
Interactive Dashboard powered by 2,46,000+ Indian Medicine Dataset
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backend import check_fraud, ocr_to_dataframe, pdf_to_dataframe

# ─────────────────────────────────────────────────
# Page Config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="ArogyX — Medicine Fraud Detector",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# CSS  — no emoji in tab selectors to avoid font glitches
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Apply Inter ONLY to text elements — never to * or it breaks icon fonts */
html, body,
h1, h2, h3, h4, h5, h6, p, span, div, label, input, button, select, textarea,
.stMarkdown, .stText, .stCaption, .stAlert, .stSuccess, .stWarning, .stInfo,
[class*="css"] { font-family: 'Inter', sans-serif; }

/* CRITICAL: keep Streamlit's own icon/symbol elements on their native font */
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
.streamlit-expanderHeader svg,
button svg,
[data-baseweb="icon"],
[data-baseweb="icon"] *,
.st-emotion-cache-p5msec,
[class*="Icon"],
[class*="icon"] { font-family: inherit !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B3C5D 0%, #1A5276 60%, #154360 100%) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #D6EAF8 !important; }
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #AED6F1 !important; }
[data-testid="stSidebar"] hr { border-color: #2E86C1 !important; }

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #0B3C5D 0%, #1A5276 50%, #117A65 100%);
    padding: 36px 40px; border-radius: 18px; margin-bottom: 28px;
    box-shadow: 0 12px 32px rgba(11,60,93,0.25);
}
.hero h1 { color: white; font-size: 2rem; font-weight: 800; margin: 0 0 6px; }
.hero p  { color: #AED6F1; font-size: 1rem; margin: 0; }

/* KPI Cards */
.kpi-card {
    background: white; border-radius: 14px;
    padding: 16px 12px 14px;
    text-align: center; box-shadow: 0 6px 18px rgba(0,0,0,0.07);
    border-top: 4px solid #1A5276; transition: transform .15s;
    min-height: 110px; box-sizing: border-box;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-label {
    font-size: 10px; font-weight: 700; color: #7F8C8D;
    text-transform: uppercase; letter-spacing: .6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    width: 100%;
}
.kpi-value {
    font-size: 1.6rem; font-weight: 800; color: #1C2833;
    margin: 4px 0 2px; line-height: 1.15;
    word-break: break-all; width: 100%;
}
.kpi-sub { font-size: 10px; color: #95A5A6; white-space: nowrap; }
.kpi-card.danger  { border-top-color: #E74C3C; }
.kpi-card.warning { border-top-color: #E67E22; }
.kpi-card.success { border-top-color: #27AE60; }
.kpi-card.info    { border-top-color: #2E86C1; }

/* Section header */
.section-title {
    font-size: 1.05rem; font-weight: 700; color: #1A5276;
    border-left: 4px solid #1A5276; padding-left: 10px;
    margin: 28px 0 14px;
}

/* Tabs — plain text, no emoji interference */
.stTabs [data-baseweb="tab-list"] {
    background: #f0f4f8; border-radius: 10px; padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important; font-size: 14px !important;
    color: #566573 !important; border-radius: 8px; padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #1A5276 !important; color: white !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: white; border-radius: 14px;
    padding: 12px; border: 2px dashed #AED6F1;
}

/* Remove Streamlit default footer */
footer { display: none !important; }

/* Custom footer */
.custom-footer {
    text-align: center; font-size: 12px; color: #95A5A6;
    padding: 24px 0 8px; border-top: 1px solid #EAECEE; margin-top: 48px;
}
.custom-footer a { color: #2E86C1; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 ArogyX")
    st.markdown("**Healthcare Billing Fraud Detector**")
    st.markdown("---")

    lang = st.selectbox("Language", ["English", "हिन्दी", "मराठी"], key="lang_select")

    st.markdown("---")
    st.markdown("### Input Method")
    input_method = st.radio(
        "input_method",
        ["Upload PDF", "Upload Image", "Camera Scan", "Manual Entry"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Detection Settings")
    fraud_threshold = st.slider(
        "Overcharge threshold (%)", min_value=1, max_value=50, value=5,
        help="Items charged more than this % above MRP are flagged as fraud",
    )
    show_not_found = st.checkbox("Show items not found in DB", value=True)

    st.markdown("---")
    st.markdown("### Database")
    st.success("246,064 active medicines loaded")
    st.caption("Indian Medicine Dataset + NIH RxNorm API")

    st.markdown("---")
    # Clear Results button — uses a flag key to trigger rerun safely
    if st.button("Clear Results", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ─────────────────────────────────────────────────
# Translations
# ─────────────────────────────────────────────────
T = {
    "English": {
        "title":     "ArogyX — Medicine Billing Fraud Detector",
        "subtitle":  "Real-time overcharge detection powered by 246,000+ Indian medicines",
        "analyzing": "Analysing your bill against 246,064 medicines…",
        "no_items":  "No valid medicine items detected. Please upload a clearer bill.",
    },
    "हिन्दी": {
        "title":     "ArogyX — दवा बिलिंग धोखाधड़ी डिटेक्टर",
        "subtitle":  "246,000+ भारतीय दवाओं से तत्काल अधिक शुल्क पहचान",
        "analyzing": "246,064 दवाओं से तुलना हो रही है…",
        "no_items":  "कोई वैध दवा नहीं मिली।",
    },
    "मराठी": {
        "title":     "ArogyX — औषध बिलिंग फसवणूक शोधक",
        "subtitle":  "246,000+ भारतीय औषधांसह तात्काळ अतिरिक्त शुल्क तपासणी",
        "analyzing": "246,064 औषधांशी तुलना होत आहे…",
        "no_items":  "वैध औषधे आढळली नाहीत.",
    },
}[lang]

# ─────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>💊 {T['title']}</h1>
    <p>{T['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Bill Input Section
# ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Submit Your Bill</div>', unsafe_allow_html=True)

bill_df = None

if input_method == "Upload PDF":
    uploaded = st.file_uploader(
        "Upload PDF bill",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a hospital or pharmacy PDF bill",
    )
    if uploaded:
        c1, c2 = st.columns([1, 6])
        with c1:
            pdf_bytes = uploaded.read()
            st.download_button("Save PDF", data=pdf_bytes,
                               file_name=uploaded.name, mime="application/pdf")
        import io
        with st.spinner(T["analyzing"]):
            bill_df = pdf_to_dataframe(io.BytesIO(pdf_bytes))
        if bill_df is not None and not bill_df.empty:
            st.success(f"Extracted {len(bill_df)} line items from PDF")

elif input_method == "Upload Image":
    uploaded = st.file_uploader(
        "Upload bill image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(uploaded, caption="Uploaded Bill", use_container_width=True)
        with col_info:
            with st.spinner(T["analyzing"]):
                bill_df = ocr_to_dataframe(uploaded)
            if bill_df is not None and not bill_df.empty:
                st.success(f"OCR extracted {len(bill_df)} line items")
                st.dataframe(bill_df, use_container_width=True, height=200)
            else:
                st.warning("Could not read items from image. Try a clearer photo.")

elif input_method == "Camera Scan":
    cam_img = st.camera_input("Point camera at your bill", label_visibility="collapsed")
    if cam_img:
        with st.spinner(T["analyzing"]):
            bill_df = ocr_to_dataframe(cam_img)
        if bill_df is not None and not bill_df.empty:
            st.success(f"Scanned {len(bill_df)} line items")

elif input_method == "Manual Entry":
    st.markdown("Enter medicine items manually:")

    if "manual_items" not in st.session_state:
        st.session_state.manual_items = [
            {"item": "Augmentin 625 Duo Tablet", "quantity": 10, "billed_mrp": 350.0},
            {"item": "Dolo 650 Tablet",          "quantity": 15, "billed_mrp": 60.0},
            {"item": "Azithral 500 Tablet",      "quantity": 5,  "billed_mrp": 200.0},
        ]

    edited = st.data_editor(
        pd.DataFrame(st.session_state.manual_items),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "item":       st.column_config.TextColumn("Medicine Name", width="large"),
            "quantity":   st.column_config.NumberColumn("Qty", min_value=1, step=1),
            "billed_mrp": st.column_config.NumberColumn("Billed Amount (Rs.)", min_value=0.0, format="%.2f"),
        },
        key="manual_editor",
    )
    st.session_state.manual_items = edited.to_dict("records")

    if st.button("Analyse Bill", type="primary", use_container_width=True):
        bill_df = edited.dropna(subset=["item"]).copy()
        bill_df = bill_df[bill_df["item"].astype(str).str.strip() != ""]

# ─────────────────────────────────────────────────
# Run fraud check when we have bill data
# ─────────────────────────────────────────────────
if bill_df is not None and not bill_df.empty:
    with st.spinner(T["analyzing"]):
        results = check_fraud(bill_df)

    # Apply sidebar threshold override
    if not results.empty and "overcharge_pct" in results.columns:
        def _apply_threshold(r):
            try:
                pct = float(r["overcharge_pct"])
                if pct > fraud_threshold:
                    return "Fraud Detected"
                return "Valid"
            except (TypeError, ValueError):
                return "MRP Not Found" if r["match_method"] == "not_found" else "Valid"

        results["status"] = results.apply(_apply_threshold, axis=1)
        results["extra_amount"] = results.apply(
            lambda r: round(
                (float(r["billed_per_unit"] or 0) - float(r["ref_per_unit"] or 0))
                * float(r["quantity"] or 1), 2
            ) if (r["ref_per_unit"] is not None
                  and str(r["ref_per_unit"]) not in ("nan", "None", ""))
            else 0.0,
            axis=1,
        )

    st.session_state["fraud_results"] = results

# ─────────────────────────────────────────────────
# Results Dashboard
# ─────────────────────────────────────────────────
if "fraud_results" in st.session_state:
    res = st.session_state["fraud_results"].copy()

    if not show_not_found:
        res = res[res["status"] != "MRP Not Found"].copy()

    if res.empty:
        st.warning(T["no_items"])
        st.stop()

    # KPI values
    total      = len(res)
    fraud_n    = int((res["status"] == "Fraud Detected").sum())
    valid_n    = int((res["status"] == "Valid").sum())
    nf_n       = int((res["status"] == "MRP Not Found").sum())
    fraud_rate = (fraud_n / total * 100) if total else 0.0
    excess     = float(res["extra_amount"].fillna(0).sum())

    # KPI cards
    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, value, sub, cls in [
        (k1, "Items Analysed",  str(total),                   f"{valid_n} valid",    "info"),
        (k2, "Fraud Detected",  str(fraud_n),                 "items overpriced",   "danger"),
        (k3, "Fraud Rate",      f"{fraud_rate:.1f}%",         "of bill",            "warning"),
        (k4, "Excess Charged",  f"₹{excess:,.0f}",            "above MRP",          "danger"),
        (k5, "Not in DB",       str(nf_n),                    "unverified items",   "info"),
    ]:
        col.markdown(
            f"<div class='kpi-card {cls}'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"<div class='kpi-sub'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs — plain text labels, no emojis (avoids font rendering bug)
    tab_summary, tab_items, tab_charts, tab_export = st.tabs([
        "Summary", "Item Analysis", "Charts", "Export",
    ])

    # ══════════════════════════════════════
    # TAB 1 — Summary
    # ══════════════════════════════════════
    with tab_summary:
        col_pie, col_sev = st.columns(2)

        with col_pie:
            status_counts = res["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_donut = px.pie(
                status_counts, names="Status", values="Count", hole=0.55,
                color="Status",
                color_discrete_map={
                    "Fraud Detected": "#E74C3C",
                    "Valid":          "#27AE60",
                    "MRP Not Found":  "#95A5A6",
                },
                title="Bill Status Breakdown",
            )
            fig_donut.update_traces(textinfo="percent+label", textfont_size=13)
            fig_donut.update_layout(
                showlegend=True, height=320,
                margin=dict(t=50, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_sev:
            sev_order = ["Critical", "Medium", "Low", "Negligible", "None", "Unknown"]
            sev_df = (
                res["overcharge_severity"]
                .value_counts()
                .reindex(sev_order)
                .dropna()
                .reset_index()
            )
            sev_df.columns = ["Severity", "Count"]
            fig_sev = px.bar(
                sev_df, x="Severity", y="Count", text="Count",
                color="Severity",
                color_discrete_map={
                    "Critical":   "#E74C3C", "Medium":     "#E67E22",
                    "Low":        "#F1C40F", "Negligible": "#27AE60",
                    "None":       "#2ECC71", "Unknown":    "#BDC3C7",
                },
                title="Overcharge Severity Distribution",
            )
            fig_sev.update_traces(textposition="outside")
            fig_sev.update_layout(showlegend=False, height=320, margin=dict(t=50, b=20))
            st.plotly_chart(fig_sev, use_container_width=True)

        # Fraud items quick-view
        fraud_items = res[res["status"] == "Fraud Detected"].sort_values(
            "overcharge_pct", ascending=False
        )
        if not fraud_items.empty:
            st.markdown('<div class="section-title">Fraud Items at a Glance</div>',
                        unsafe_allow_html=True)
            for _, row in fraud_items.iterrows():
                pct   = float(row.get("overcharge_pct") or 0)
                extra = float(row.get("extra_amount") or 0)
                fa, fb, fc, fd = st.columns([3, 2, 2, 2])
                fa.markdown(
                    f"**{str(row['item']).title()}**  \n"
                    f"<span style='color:#7F8C8D;font-size:12px'>"
                    f"Matched: {row.get('matched_name','—')}</span>",
                    unsafe_allow_html=True,
                )
                fb.metric("Billed/unit",   f"Rs.{float(row.get('billed_per_unit') or 0):.2f}")
                fc.metric("Ref MRP/unit",  f"Rs.{float(row.get('ref_per_unit') or 0):.2f}")
                fd.metric("Overcharge",    f"{pct:.1f}%",
                          delta=f"Rs.{extra:.2f} extra", delta_color="inverse")
                st.divider()

    # ══════════════════════════════════════
    # TAB 2 — Item Analysis
    # ══════════════════════════════════════
    with tab_items:
        fc1, fc2, fc3 = st.columns([3, 2, 2])
        with fc1:
            search = st.text_input("Search medicine name", placeholder="e.g. Augmentin")
        with fc2:
            status_filter = st.multiselect(
                "Filter by Status",
                ["Fraud Detected", "Valid", "MRP Not Found"],
                default=["Fraud Detected", "Valid", "MRP Not Found"],
            )
        with fc3:
            method_filter = st.multiselect(
                "Filter by Match Method",
                ["exact", "fuzzy", "rxnorm", "ingredient", "not_found"],
                default=["exact", "fuzzy", "rxnorm", "ingredient", "not_found"],
            )

        disp = res.copy()
        if search:
            mask = (
                disp["item"].str.contains(search, case=False, na=False)
                | disp["matched_name"].fillna("").str.contains(search, case=False, na=False)
            )
            disp = disp[mask]
        if status_filter:
            disp = disp[disp["status"].isin(status_filter)]
        if method_filter:
            disp = disp[disp["match_method"].isin(method_filter)]

        st.caption(f"Showing {len(disp)} of {len(res)} items")

        table = disp[[
            "item", "quantity", "billed_mrp",
            "matched_name", "composition",
            "billed_per_unit", "ref_per_unit",
            "overcharge_pct", "overcharge_severity",
            "match_score", "match_method", "status",
        ]].copy()
        table.columns = [
            "Billed Item", "Qty", "Billed Total",
            "Matched Medicine", "Composition",
            "Billed/Unit", "Ref MRP/Unit",
            "Overcharge %", "Severity",
            "Match Score", "Match Method", "Status",
        ]

        def _col_status(v):
            c = {"Fraud Detected": "#FADBD8", "Valid": "#EAFAF1", "MRP Not Found": "#F2F3F4"}
            return f"background-color:{c.get(v,'')};font-weight:600"

        def _col_oc(v):
            try:
                f = float(v)
                if f > 50: return "background-color:#FADBD8;color:#C0392B;font-weight:700"
                if f > 20: return "background-color:#FDEBD0;color:#CA6F1E;font-weight:700"
                if f > 5:  return "background-color:#FEF9E7;color:#B7950B;font-weight:700"
                if f > 0:  return "background-color:#EAFAF1;color:#1E8449"
            except (TypeError, ValueError):
                pass
            return ""

        def _fmt(x, suffix=""):
            try:
                return f"{float(x):.2f}{suffix}" if x and str(x) not in ("nan","None") else "—"
            except (TypeError, ValueError):
                return "—"

        styled = (
            table.style
            .applymap(_col_status, subset=["Status"])
            .applymap(_col_oc,     subset=["Overcharge %"])
            .format({
                "Billed Total":  lambda x: _fmt(x),
                "Billed/Unit":   lambda x: _fmt(x),
                "Ref MRP/Unit":  lambda x: _fmt(x),
                "Overcharge %":  lambda x: _fmt(x, "%"),
                "Match Score":   lambda x: f"{float(x):.0f}" if x and str(x) not in ("nan","None") else "—",
            }, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True, height=440)

        # Expandable detail per fraud item
        fraud_disp = disp[disp["status"] == "Fraud Detected"]
        if not fraud_disp.empty:
            st.markdown('<div class="section-title">Fraud Item Details</div>',
                        unsafe_allow_html=True)
            for _, row in fraud_disp.iterrows():
                item_name = str(row["item"]).title()
                try:
                    pct_val  = float(row.get("overcharge_pct") or 0)
                    exp_title = f"{item_name}  —  {pct_val:.1f}% overcharge"
                except (TypeError, ValueError):
                    exp_title = item_name

                with st.expander(exp_title):
                    d1, d2, d3 = st.columns(3)
                    d1.markdown(f"**Matched Name**  \n{row.get('matched_name') or '—'}")
                    d1.markdown(f"**Manufacturer**  \n{row.get('manufacturer') or '—'}")
                    d1.markdown(f"**Pack Size**  \n{row.get('pack_size') or '—'}")
                    d2.markdown(f"**Composition**  \n{row.get('composition') or '—'}")
                    d2.markdown(f"**Match Method**  \n`{row.get('match_method') or '—'}`")
                    d2.markdown(f"**Match Score**  \n{row.get('match_score') or '—'}")
                    d3.metric("Billed/unit",   f"Rs.{float(row.get('billed_per_unit') or 0):.2f}")
                    d3.metric("Ref MRP/unit",  f"Rs.{float(row.get('ref_per_unit') or 0):.2f}")
                    d3.metric("Excess Amount", f"Rs.{float(row.get('extra_amount') or 0):.2f}")

    # ══════════════════════════════════════
    # TAB 3 — Charts
    # ══════════════════════════════════════
    with tab_charts:
        sev_colour_map = {
            "Critical": "#E74C3C", "Medium": "#E67E22",
            "Low": "#F1C40F",      "Negligible": "#27AE60", "None": "#2ECC71",
        }

        # Billed vs Reference grouped bar
        price_df = res.dropna(subset=["ref_per_unit"]).copy()
        if not price_df.empty:
            price_df["label"] = price_df["item"].str.title()
            fig_cmp = go.Figure([
                go.Bar(
                    name="Reference MRP (per unit)",
                    x=price_df["label"], y=price_df["ref_per_unit"],
                    marker_color="#27AE60",
                    text=price_df["ref_per_unit"].apply(lambda x: f"Rs.{x:.2f}"),
                    textposition="outside",
                ),
                go.Bar(
                    name="Billed Price (per unit)",
                    x=price_df["label"], y=price_df["billed_per_unit"],
                    marker_color="#E74C3C",
                    text=price_df["billed_per_unit"].apply(lambda x: f"Rs.{x:.2f}"),
                    textposition="outside",
                ),
            ])
            fig_cmp.update_layout(
                title="Billed Price vs Reference MRP per Unit",
                barmode="group", yaxis_title="Rs. per unit",
                xaxis_tickangle=-35,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=400, margin=dict(t=80, b=60),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        # Overcharge % horizontal bar (fraud only)
        oc_df = res[res["status"] == "Fraud Detected"].dropna(subset=["overcharge_pct"]).copy()
        if not oc_df.empty:
            oc_df = oc_df.sort_values("overcharge_pct")
            oc_df["label"] = oc_df["item"].str.title()
            fig_oc = px.bar(
                oc_df, y="label", x="overcharge_pct", orientation="h",
                color="overcharge_severity", color_discrete_map=sev_colour_map,
                title="Overcharge % by Medicine",
                labels={"label": "Medicine", "overcharge_pct": "Overcharge %"},
                text=oc_df["overcharge_pct"].apply(lambda x: f"{x:.1f}%"),
            )
            fig_oc.update_traces(textposition="outside")
            fig_oc.update_layout(
                height=max(300, len(oc_df) * 50),
                margin=dict(t=60, b=20, l=20, r=60),
            )
            st.plotly_chart(fig_oc, use_container_width=True)

        # Excess amount bar
        exc_df = res[res["extra_amount"].fillna(0) > 0].copy()
        if not exc_df.empty:
            exc_df = exc_df.sort_values("extra_amount", ascending=False)
            exc_df["label"] = exc_df["item"].str.title()
            fig_ex = px.bar(
                exc_df, x="label", y="extra_amount",
                color="overcharge_severity", color_discrete_map=sev_colour_map,
                title="Excess Amount Charged per Medicine (Rs.)",
                labels={"label": "Medicine", "extra_amount": "Excess (Rs.)"},
                text=exc_df["extra_amount"].apply(lambda x: f"Rs.{x:.2f}"),
            )
            fig_ex.update_traces(textposition="outside")
            fig_ex.update_layout(height=360, margin=dict(t=60, b=60), xaxis_tickangle=-30)
            st.plotly_chart(fig_ex, use_container_width=True)

        # Match method donut
        mm_df = res["match_method"].value_counts().reset_index()
        mm_df.columns = ["Method", "Count"]
        fig_mm = px.pie(
            mm_df, names="Method", values="Count", hole=0.45,
            color="Method",
            color_discrete_map={
                "exact": "#27AE60", "fuzzy": "#2E86C1",
                "rxnorm": "#8E44AD", "ingredient": "#E67E22", "not_found": "#95A5A6",
            },
            title="Match Method Distribution",
        )
        fig_mm.update_layout(height=320, margin=dict(t=50, b=10))
        st.plotly_chart(fig_mm, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 4 — Export
    # ══════════════════════════════════════
    with tab_export:
        st.markdown('<div class="section-title">Download Analysis</div>',
                    unsafe_allow_html=True)

        export_df = res[[
            "item", "quantity", "billed_mrp",
            "matched_name", "composition", "pack_size", "manufacturer",
            "billed_per_unit", "ref_per_unit",
            "overcharge_pct", "overcharge_severity",
            "extra_amount", "match_score", "match_method", "status",
        ]].copy()
        export_df.columns = [
            "Billed Item", "Quantity", "Billed Total",
            "Matched Medicine", "Composition", "Pack Size", "Manufacturer",
            "Billed/Unit", "Ref MRP/Unit",
            "Overcharge %", "Severity",
            "Excess Amount", "Match Score", "Match Method", "Status",
        ]

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download as CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="arogyx_fraud_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            fraud_only   = export_df[export_df["Status"] == "Fraud Detected"]
            total_excess = res["extra_amount"].fillna(0).sum()
            html_report  = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ArogyX Fraud Report</title>
<style>
  body{{font-family:Arial,sans-serif;padding:30px;background:#f9f9f9;}}
  h1{{color:#1A5276;}} h2{{color:#E74C3C;}}
  table{{border-collapse:collapse;width:100%;background:white;}}
  th{{background:#1A5276;color:white;padding:10px;text-align:left;}}
  td{{padding:8px 10px;border-bottom:1px solid #eee;}}
  tr:hover{{background:#f5f5f5;}}
</style></head><body>
<h1>ArogyX - Fraud Detection Report</h1>
<p><b>Total Items:</b> {total} | <b>Fraud:</b> {fraud_n} | <b>Excess:</b> Rs.{total_excess:,.2f}</p>
<h2>Overcharged Items</h2>
{fraud_only.to_html(index=False, border=0)}
<br><p style="color:#95A5A6;font-size:12px;">
ArogyX | Indian Medicine Dataset | NIH RxNorm API</p>
</body></html>"""
            st.download_button(
                "Download HTML Report",
                data=html_report.encode("utf-8"),
                file_name="arogyx_fraud_report.html",
                mime="text/html",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("**Preview:**")
        st.dataframe(export_df, use_container_width=True, height=300)

        with st.expander("Match Method Guide"):
            st.markdown("""
| Method | Meaning |
|---|---|
| `exact` | Name matched exactly after cleaning |
| `fuzzy` | Matched by similarity (score >= 75) |
| `rxnorm` | Brand resolved to generic via NIH RxNorm API |
| `ingredient` | Matched via active ingredient in composition |
| `not_found` | No match in 246,064 medicine database |

**Severity thresholds:**
- Critical: > 50% above MRP
- Medium: 20-50% above MRP
- Low: 5-20% above MRP
- Negligible: < 5% above MRP
""")

# ─────────────────────────────────────────────────
# No results yet — landing state
# ─────────────────────────────────────────────────
else:
    st.info(
        "Upload a bill or use Manual Entry to start. "
        "The app compares each medicine against 246,064 real Indian medicines "
        "and flags any overcharging.",
        icon="💡",
    )

    sample_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "sample_bill.csv"
    )
    if os.path.exists(sample_path):
        with st.expander("Run Demo with Sample Bill"):
            sample_df = pd.read_csv(sample_path)
            st.dataframe(sample_df, use_container_width=True)
            if st.button("Analyse Sample Bill", type="primary"):
                with st.spinner("Analysing sample bill…"):
                    st.session_state["fraud_results"] = check_fraud(sample_df)
                st.rerun()

# ─────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    ArogyX — Government of India · Digital Health Intelligence Platform<br>
    <small>Medicine data:
    <a href="https://github.com/junioralive/Indian-Medicine-Dataset" target="_blank">
    Indian Medicine Dataset (246,064 drugs)</a> ·
    Name normalisation:
    <a href="https://rxnav.nlm.nih.gov" target="_blank">NIH RxNorm API</a>
    </small>
</div>
""", unsafe_allow_html=True)

from pathlib import Path
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st 


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "images"

SENTIMENT_COLORS = {
    "Negatif": "#ef4444",
    "Netral": "#94a3b8",
    "Positif": "#2563eb",
}

TOPIC_SUMMARY = [
    {
        "title": "Paket Hilang, Rusak, dan Refund",
        "count": "1.238 tweet",
        "description": "Paket rusak atau hilang, refund lambat, uang tertahan, dan kecewa pada layanan brand.",
        "keywords": ["hilang", "paket", "rusak", "telat", "refund", "uang", "kecewa", "brand"],
    },
    {
        "title": "Promo, Ongkir, dan Pengiriman",
        "count": "97 tweet",
        "description": "Diskusi promo aplikasi, ongkir, diskon, dan aktivitas event belanja.",
        "keywords": ["kirim", "paket", "ongkir", "diskon", "aplikasi", "promo", "spesial"],
    },
    {
        "title": "Kurir, Resi, dan Status Barang",
        "count": "570 tweet",
        "description": "Keluhan operasional tentang kurir, resi, status pengiriman, dan respons bantuan.",
        "keywords": ["kurir", "resi", "barang", "hilang", "dikirim", "kak", "gratis"],
    },
]

MODEL_COMPARISON = [
    {"model": "XGBoost", "accuracy": "0.8268", "f1_macro": "0.4692", "f1_weighted": "0.8081"},
    {"model": "SVM", "accuracy": "0.7874", "f1_macro": "0.4742", "f1_weighted": "0.7930"},
    {"model": "Naive Bayes", "accuracy": "0.8268", "f1_macro": "0.4246", "f1_weighted": "0.7850"},
    {"model": "KNN", "accuracy": "0.7402", "f1_macro": "0.4334", "f1_weighted": "0.7559"},
]

EVIDENCE_KEYWORDS = [
    "hilang",
    "rusak",
    "refund",
    "telat",
    "lambat",
    "kurir",
    "resi",
    "paket",
    "barang",
    "komplain",
    "kecewa",
    "cctv",
]

COMPLAINT_EVIDENCE_KEYWORDS = [
    "paket hilang",
    "paket rusak",
    "barang hilang",
    "barang rusak",
    "pesanan hilang",
    "pesanan rusak",
    "belum sampai",
    "gak sampai",
    "ga sampai",
    "tidak sampai",
    "gadateng",
    "gak dateng",
    "ga dateng",
    "tidak datang",
    "telat",
    "terlambat",
    "refund",
    "pengembalian dana",
    "retur",
    "diretur",
    "uang",
    "kurir",
    "resi",
    "cod",
    "komplain",
    "keluhan",
    "kecewa",
    "lambat",
    "nyangkut",
    "status",
    "dikirim",
    "pengiriman",
    "salah sortir",
    "nyasar",
]

EVIDENCE_CONTEXT_KEYWORDS = [
    "paket",
    "barang",
    "pesanan",
    "order",
    "checkout",
    "resi",
    "kurir",
    "refund",
    "retur",
    "dana",
    "cod",
    "ongkir",
    "ekspedisi",
    "pengiriman",
    "dikirim",
    "shopee",
    "lazada",
    "tokopedia",
    "tiktok shop",
    "jnt",
    "jne",
    "sicepat",
    "anteraja",
    "ninja",
]

EVIDENCE_NOISE_KEYWORDS = [
    "giveaway",
    "alert",
    "promo",
    "voucher",
    "diskon",
    "gratis ongkir",
    "flash sale",
    "spesial",
]

EVIDENCE_EXCLUDED_ACCOUNTS = {
    "detikcom",
    "jakmall",
    "lazadaid",
    "jntexpressid",
    "kompascom",
}


st.set_page_config(
    page_title="[5] Dashboard War Kredibilitas Brand 'E-Commerce & Ekspedisi Logistik'",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "hasil_utama.csv")
    issues = pd.read_csv(DATA_DIR / "issue.csv")
    accounts = pd.read_csv(DATA_DIR / "top_accounts.csv")
    edges = pd.read_csv(DATA_DIR / "edge_list.csv")
    return df, issues, accounts, edges


df, issue_df, top_accounts_df, edges_df = load_data()

for column in ["brand", "kategori", "Sentiment", "topic_label", "username", "full_text"]:
    if column in df.columns:
        df[column] = df[column].fillna("-")


def fmt_int(value):
    return f"{int(value):,}".replace(",", ".")


def fmt_pct(value):
    return f"{value:.1f}%"


def short_text(value, limit=135):
    text = " ".join(str(value).split())
    return f"{text[:limit]}..." if len(text) > limit else text


def section_title(eyebrow, title, anchor=None):
    anchor_html = f" id='{anchor}'" if anchor else ""
    st.markdown(
        f"""
        <div{anchor_html} class="section-head">
            <div class="eyebrow">{eyebrow}</div>
            <h2>{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, primary=False):
    kind = " primary" if primary else ""
    st.markdown(
        f"""
        <div class="metric-card{kind}">
            <span>{label}</span>
            <strong>{value}</strong>
            <p>{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_image(path):
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"Gambar belum tersedia: {path.name}")


def chart_config():
    return {"displayModeBar": False, "responsive": True}


def make_brand_chart(data):
    grouped = (
        data.groupby(["kategori", "brand", "Sentiment"])
        .size()
        .reset_index(name="Jumlah")
    )
    fig = px.bar(
        grouped,
        x="brand",
        y="Jumlah",
        color="Sentiment",
        facet_col="kategori",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"Sentiment": ["Negatif", "Netral", "Positif"]},
    )
    fig.for_each_annotation(lambda item: item.update(text=item.text.split("=")[-1]))
    fig.update_layout(
        height=430,
        margin=dict(l=8, r=8, t=42, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        legend_title_text="",
        legend=dict(orientation="h", y=-0.14, x=0.31),
        font=dict(color="#596a62", size=13),
        bargap=0.12,
        bargroupgap=0.02,
    )
    fig.update_xaxes(title=None, tickangle=0)
    fig.update_xaxes(matches=None)
    fig.update_yaxes(title=None, gridcolor="#edf1ed")
    return fig


def make_sentiment_chart(data):
    sentiment = (
        data["Sentiment"]
        .value_counts()
        .reindex(["Positif", "Netral", "Negatif"])
        .fillna(0)
        .reset_index()
    )
    sentiment.columns = ["Sentiment", "Jumlah"]
    fig = px.pie(
        sentiment,
        names="Sentiment",
        values="Jumlah",
        hole=0.62,
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(
        height=230,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="#596a62", size=12),
    )
    return fig, sentiment


def make_issue_chart(data, issues):
    filtered_issues = issues.copy()
    if set(data["kategori"].unique()) != set(df["kategori"].unique()):
        filtered_issues = filtered_issues[filtered_issues["Kategori"].isin(data["kategori"].unique())]
    fig = px.bar(
        filtered_issues.sort_values("Jumlah", ascending=True),
        x="Jumlah",
        y="Isu",
        color="Kategori",
        orientation="h",
        color_discrete_map={"E-Commerce": "#1d4ed8", "Ekspedisi": "#60a5fa"},
    )
    fig.update_layout(
        height=270,
        margin=dict(l=8, r=8, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        legend_title_text="",
        legend=dict(orientation="h", y=1.22, x=0.52),
        font=dict(color="#596a62", size=12),
    )
    fig.update_xaxes(title=None, gridcolor="#edf1ed")
    fig.update_yaxes(title=None)
    return fig


def make_evidence_tweets(data, limit=8):
    negative = data[data["Sentiment"].eq("Negatif")].copy()
    if negative.empty:
        return negative

    topic_text = negative["topic_label"] if "topic_label" in negative.columns else pd.Series("", index=negative.index)
    clean_text = negative["clean_text"] if "clean_text" in negative.columns else pd.Series("", index=negative.index)
    text_source = (
        negative["full_text"].astype(str)
        + " "
        + clean_text.astype(str)
        + " "
        + topic_text.astype(str)
    ).str.lower()
    negative["_theme_score"] = text_source.map(
        lambda text: (
            sum(2 for keyword in COMPLAINT_EVIDENCE_KEYWORDS if keyword in text)
            + sum(keyword in text for keyword in EVIDENCE_KEYWORDS)
        )
    )
    negative["_context_score"] = text_source.map(
        lambda text: sum(keyword in text for keyword in EVIDENCE_CONTEXT_KEYWORDS)
    )
    negative["_noise_score"] = text_source.map(
        lambda text: sum(keyword in text for keyword in EVIDENCE_NOISE_KEYWORDS)
    )
    favorite_count = (
        negative["favorite_count"]
        if "favorite_count" in negative.columns
        else pd.Series(0, index=negative.index)
    )
    negative["_favorite_count"] = pd.to_numeric(favorite_count, errors="coerce").fillna(0)
    negative["_username_key"] = negative["username"].astype(str).str.lower().str.lstrip("@")

    themed = negative[
        negative["_theme_score"].ge(2)
        & negative["_context_score"].gt(0)
        & negative["_noise_score"].eq(0)
        & ~negative["_username_key"].isin(EVIDENCE_EXCLUDED_ACCOUNTS)
    ]
    if themed.empty:
        themed = negative[negative["_theme_score"].gt(0)]
    pool = themed if not themed.empty else negative
    category_order = ["E-Commerce", "Ekspedisi"]
    categories = [category for category in category_order if category in set(pool["kategori"].dropna())]
    categories.extend(
        category for category in sorted(pool["kategori"].dropna().unique()) if category not in categories
    )
    if not categories:
        return pool.sort_values(["_theme_score", "_favorite_count"], ascending=False).head(limit)

    ranked_by_category = {
        category: (
            pool[pool["kategori"].eq(category)]
            .sort_values(["_theme_score", "_favorite_count"], ascending=False)
            .index.tolist()
        )
        for category in categories
    }
    selected_indexes = []

    while len(selected_indexes) < limit and any(ranked_by_category.values()):
        for category in categories:
            if ranked_by_category[category]:
                selected_indexes.append(ranked_by_category[category].pop(0))
            if len(selected_indexes) >= limit:
                break

    if len(selected_indexes) < limit:
        remaining = (
            pool[~pool.index.isin(selected_indexes)]
            .sort_values(["_theme_score", "_favorite_count"], ascending=False)
            .head(limit - len(selected_indexes))
        )
        selected_indexes.extend(remaining.index.tolist())

    return pool.loc[selected_indexes].head(limit)


def render_model_comparison_table():
    rows = []
    for item in MODEL_COMPARISON:
        best_badge = "<span class='best-pill'>Terpilih</span>" if item["model"] == "XGBoost" else ""
        row_class = " best-model" if item["model"] == "XGBoost" else ""
        rows.append(
            f'<div class="model-row{row_class}">'
            f'<div>{item["model"]}{best_badge}</div>'
            f'<div>{item["accuracy"]}</div>'
            f'<div>{item["f1_macro"]}</div>'
            f'<div>{item["f1_weighted"]}</div>'
            "</div>"
        )

    return (
        '<div class="model-compare">'
        '<div class="model-row model-head">'
        "<div>Model</div>"
        "<div>Accuracy</div>"
        "<div>F1 Macro</div>"
        "<div>F1 Weighted</div>"
        "</div>"
        f'<div class="model-body">{"".join(rows)}</div>'
        "</div>"
    )


def enrich_accounts(accounts, edges):
    edge_stats = {}
    for _, edge in edges.iterrows():
        for account in [edge.get("Source"), edge.get("Target")]:
            if pd.isna(account) or not account:
                continue
            key = str(account).lower()
            stats = edge_stats.setdefault(key, {"relations": 0, "brands": set(), "categories": set()})
            stats["relations"] += 1
            if pd.notna(edge.get("Brand")):
                stats["brands"].add(edge.get("Brand"))
            if pd.notna(edge.get("Kategori")):
                stats["categories"].add(edge.get("Kategori"))

    enriched = accounts.sort_values("Betweenness", ascending=False).copy()
    keys = enriched["Akun"].str.lower()
    enriched["Relasi"] = keys.map(lambda key: edge_stats.get(key, {}).get("relations", 0))
    enriched["Brand"] = keys.map(lambda key: ", ".join(sorted(edge_stats.get(key, {}).get("brands", []))) or "-")
    enriched["Kategori"] = keys.map(lambda key: ", ".join(sorted(edge_stats.get(key, {}).get("categories", []))) or "-")
    enriched["Urgensi"] = [
        "Tinggi" if index < 3 and row.Betweenness > 0 else "Sedang" if index < 10 and row.Betweenness > 0 else "Monitor"
        for index, row in enumerate(enriched.itertuples())
    ]
    enriched["Betweenness"] = enriched["Betweenness"].map(lambda value: f"{value:.6f}")
    enriched["Akun"] = "@" + enriched["Akun"].astype(str)
    return enriched


def render_centrality_table(accounts):
    urgency_class = {
        "Tinggi": "high",
        "Sedang": "medium",
        "Monitor": "low",
    }
    rows = []
    for _, row in accounts.iterrows():
        urgency = str(row["Urgensi"])
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['Akun']))}</td>"
            f"<td>{escape(str(row['Brand']))}</td>"
            f"<td>{escape(str(row['Relasi']))}</td>"
            f"<td>{escape(str(row['Betweenness']))}</td>"
            f"<td><span class='urgency-pill {urgency_class.get(urgency, 'low')}'>{escape(urgency)}</span></td>"
            "</tr>"
        )

    return (
        '<div class="centrality-table-wrap">'
        '<table class="centrality-table">'
        "<thead><tr>"
        "<th>Akun</th>"
        "<th>Brand</th>"
        "<th>Relasi</th>"
        "<th>Betweenness</th>"
        "<th>Urgensi</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


st.markdown(
    """
    <style>
    :root{
        --bg:#eef4ff;
        --panel:#ffffff;
        --text:#172033;
        --muted:#66758a;
        --line:#dbe7f6;
        --green:#2563eb;
        --green-dark:#1d4ed8;
        --green-soft:#dbeafe;
        --shadow:0 18px 42px rgba(30,64,175,.10);
    }

    .stApp{
        background:linear-gradient(140deg,#eef4ff 0%,#f8fbff 44%,#edf6ff 100%);
    }

    html, body, [class*="css"]{
        font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color:var(--text);
    }

    .main .block-container{
        max-width:1220px;
        padding:18px 22px 44px;
    }

    header[data-testid="stHeader"]{
        background:transparent;
    }

    section[data-testid="stSidebar"]{
        background:transparent;
    }

    section[data-testid="stSidebar"] > div{
        background:#fff;
        border:1px solid rgba(255,255,255,.95);
        border-radius:30px;
        box-shadow:var(--shadow);
        margin:18px 10px 18px 18px;
        padding:20px 16px;
    }

    .sidebar-brand{
        display:flex;
        align-items:center;
        gap:14px;
        margin:4px 0 18px;
    }

    .sidebar-logo{
        width:48px;
        height:48px;
        border-radius:16px;
        display:grid;
        place-items:center;
        background:var(--green);
        color:#fff;
        font-weight:900;
        box-shadow:0 14px 28px rgba(37,99,235,.22);
    }

    .sidebar-title{
        font-size:21px;
        font-weight:900;
        letter-spacing:-.03em;
        line-height:1.05;
    }

    .sidebar-subtitle{
        color:#7c8b84;
        font-size:13px;
        margin-top:4px;
    }

    .nav-link{
        display:block;
        padding:8px 12px;
        border-radius:14px;
        color:#53665d !important;
        text-decoration:none !important;
        font-weight:800;
        margin-bottom:4px;
    }

    .nav-link:hover{
        background:#eff6ff;
        color:var(--green-dark) !important;
    }

    .stApp:has(#dashboard-analytics:target) .nav-link[href="#dashboard-analytics"],
    .stApp:has(#brand-analysis:target) .nav-link[href="#brand-analysis"],
    .stApp:has(#top-issue:target) .nav-link[href="#top-issue"],
    .stApp:has(#topic-modeling:target) .nav-link[href="#topic-modeling"],
    .stApp:has(#wordcloud:target) .nav-link[href="#wordcloud"],
    .stApp:has(#social-network-analysis:target) .nav-link[href="#social-network-analysis"],
    .stApp:has(#data-tweet:target) .nav-link[href="#data-tweet"]{
        background:#eff6ff;
        color:var(--green-dark) !important;
    }

    .side-note{
        margin-top:14px;
        padding:14px;
        border-radius:20px;
        background:#eff6ff;
        border:1px solid #dbeafe;
        color:#63756d;
        line-height:1.45;
        font-size:13px;
    }

    .side-note strong{
        display:block;
        color:var(--green-dark);
        font-size:15px;
        margin-bottom:6px;
    }

    .search-with-icon + div[data-testid="stTextInput"]{
        position:relative;
    }

    .search-with-icon + div[data-testid="stTextInput"] input{
        padding-right:42px;
    }

    .search-with-icon + div[data-testid="stTextInput"]::after{
        content:"";
        position:absolute;
        right:14px;
        bottom:14px;
        width:18px;
        height:18px;
        pointer-events:none;
        background-color:#94a3b8;
        -webkit-mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E") center / contain no-repeat;
        mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E") center / contain no-repeat;
    }

    .hero-card,
    .metric-card{
        background:#fff;
        border:1px solid rgba(255,255,255,.95);
        border-radius:26px;
        box-shadow:var(--shadow);
    }

    .hero-card{
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:18px;
        padding:22px 26px;
        margin-bottom:14px;
    }

    .hero-card h1{
        margin:6px 0 8px;
        color:var(--text);
        font-size:32px;
        font-weight:900;
        line-height:1.12;
        letter-spacing:-.035em;
    }

    .hero-card p{
        margin:0;
        color:var(--muted);
        font-size:15px;
        font-weight:600;
    }

    .dataset-pill{
        flex:0 0 auto;
        padding:13px 17px;
        border-radius:999px;
        background:#f8fbff;
        border:1px solid var(--line);
        color:#40554b;
        font-size:13px;
        font-weight:900;
    }

    .metric-card{
        min-height:112px;
        padding:16px 18px;
        position:relative;
        overflow:hidden;
    }

    .metric-card:after{
        content:"";
        position:absolute;
        right:-36px;
        top:-36px;
        width:92px;
        height:92px;
        border-radius:50%;
        background:var(--green-soft);
    }

    .metric-card.primary{
        background:linear-gradient(145deg,#2563eb 0%,#1d4ed8 100%);
        color:#fff;
    }

    .metric-card.primary:after{
        background:rgba(255,255,255,.17);
    }

    .metric-card span{
        display:block;
        color:#6b7b73;
        font-size:12px;
        font-weight:900;
        margin-bottom:12px;
    }

    .metric-card.primary span,
    .metric-card.primary p{
        color:#eff6ff;
    }

    .metric-card strong{
        display:block;
        font-size:29px;
        font-weight:900;
        line-height:1;
        letter-spacing:-.05em;
        margin-bottom:9px;
        color:inherit;
    }

    .metric-card p{
        margin:0;
        color:var(--green-dark);
        font-size:13px;
        line-height:1.35;
        font-weight:900;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]{
        border:1px solid rgba(255,255,255,.94);
        border-radius:22px;
        background:#fff;
        box-shadow:var(--shadow);
        height:auto !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div{
        padding:16px 18px;
        height:auto !important;
        min-height:0 !important;
    }

    div[data-testid="stHorizontalBlock"]{
        align-items:flex-start;
        gap:1rem;
    }

    .section-head{
        margin-bottom:10px;
    }

    .eyebrow{
        color:var(--green);
        font-size:12px;
        font-weight:900;
        letter-spacing:.14em;
        text-transform:uppercase;
        margin-bottom:6px;
    }

    .modeling-title{
        color:#111827;
        font-size:26px;
        font-weight:900;
        letter-spacing:0;
        text-transform:none;
        line-height:1.18;
        margin:0 0 14px;
    }

    .section-head h2{
        margin:0;
        color:var(--text);
        font-size:26px;
        font-weight:900;
        line-height:1.18;
        letter-spacing:-.035em;
    }

    .topic-panel{
        padding:18px;
        border-radius:22px;
        background:linear-gradient(145deg,#2563eb 0%,#1e40af 100%);
        color:#fff;
        box-shadow:var(--shadow);
        margin-bottom:12px;
    }

    .score-row{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        margin-bottom:18px;
    }

    .score-label{
        color:#dbeafe;
        font-size:12px;
        font-weight:900;
        letter-spacing:.14em;
        text-transform:uppercase;
    }

    .score-pill{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        min-width:104px;
        padding:8px 14px;
        border:1px solid rgba(255,255,255,.46);
        border-radius:999px;
        background:rgba(255,255,255,.14);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.25);
        color:#fff;
        font-size:24px;
        font-weight:900 !important;
        line-height:1;
        backdrop-filter:blur(10px);
    }

    .topic-panel h2{
        margin:0;
        color:#fff;
        font-size:24px;
        font-weight:950;
        line-height:1.12;
        letter-spacing:-.02em;
    }

    .topic-card{
        padding:14px;
        border-radius:18px;
        background:#fff;
        color:var(--text);
        border:1px solid rgba(255,255,255,.92);
        box-shadow:0 10px 24px rgba(30,64,175,.08);
        margin-bottom:10px;
    }

    .topic-card h3{
        margin:0 0 8px;
        font-size:18px;
        font-weight:900;
        line-height:1.22;
    }

    .topic-card p{
        margin:0 0 10px;
        color:#64756d;
        font-size:14px;
        font-weight:600;
        line-height:1.45;
    }

    .topic-count{
        float:right;
        padding:5px 8px;
        border-radius:999px;
        background:#f1f5f9;
        border:1px solid #e2e8f0;
        color:#475569;
        font-size:11px;
        font-weight:900;
        margin-left:8px;
    }

    .tag{
        display:inline-block;
        margin:3px 3px 0 0;
        padding:6px 9px;
        border-radius:999px;
        background:var(--green-soft);
        color:var(--green-dark);
        border:1px solid transparent;
        font-size:11px;
        font-weight:900;
    }

    .tag.cluster-0{
        background:#dff3ed;
        border-color:#b9e3d6;
        color:#23856d;
    }

    .tag.cluster-1{
        background:#ffe8dc;
        border-color:#ffc8ad;
        color:#c4512c;
    }

    .tag.cluster-2{
        background:#e6ebf8;
        border-color:#cbd5ee;
        color:#5069a8;
    }

    .summary-grid{
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:10px;
        margin:10px 0;
    }

    .summary-grid.two{
        grid-template-columns:minmax(0, 230px);
        width:min(100%, 230px);
    }

    .summary-box{
        padding:12px;
        border-radius:16px;
        border:1px solid var(--line);
        background:#f8fbff;
    }

    .summary-box span{
        display:block;
        margin-bottom:8px;
        color:#65736c;
        font-size:12px;
        font-weight:900;
        text-transform:uppercase;
        letter-spacing:.06em;
    }

    .summary-box strong{
        display:block;
        color:var(--text);
        font-size:19px;
        font-weight:900;
        line-height:1.15;
        overflow-wrap:anywhere;
    }

    .model-compare{
        width:min(100%, 620px);
        margin-top:14px;
        overflow:hidden;
        border:1px solid var(--line);
        border-radius:16px;
        background:#fff;
        font-size:13px;
    }

    .model-row{
        display:grid;
        grid-template-columns:1.25fr .8fr .8fr .9fr;
        align-items:center;
    }

    .model-row > div{
        padding:10px 12px;
        border-bottom:1px solid var(--line);
    }

    .model-head > div{
        background:#f8fbff;
        color:#64756d;
        font-size:11px;
        font-weight:900;
        letter-spacing:.08em;
        text-transform:uppercase;
    }

    .model-body > .model-row > div{
        color:var(--text);
        font-weight:800;
    }

    .model-body > .model-row:last-child > div{
        border-bottom:0;
    }

    .model-row.best-model > div{
        background:#f8fafc;
    }

    .best-pill{
        display:inline-block;
        margin-left:6px;
        padding:3px 7px;
        border-radius:999px;
        background:#dbeafe;
        border:1px solid #bfdbfe;
        color:#1d4ed8;
        font-size:10px;
        font-weight:900;
        letter-spacing:.04em;
    }

    .centrality-table-wrap{
        height:360px;
        overflow:auto;
        border:1px solid var(--line);
        border-radius:16px;
        background:#fff;
    }

    .centrality-table{
        width:100%;
        border-collapse:separate;
        border-spacing:0;
        font-size:14px;
    }

    .centrality-table th,
    .centrality-table td{
        padding:11px 12px;
        border-bottom:1px solid var(--line);
        text-align:left;
        color:var(--text);
        vertical-align:middle;
    }

    .centrality-table th{
        position:sticky;
        top:0;
        z-index:1;
        background:#f8fbff;
        color:#64756d;
        font-size:11px;
        font-weight:900;
        letter-spacing:.08em;
        text-transform:uppercase;
    }

    .centrality-table tbody tr:last-child td{
        border-bottom:0;
    }

    .urgency-pill{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        min-width:72px;
        padding:5px 10px;
        border-radius:999px;
        border:1px solid transparent;
        font-size:12px;
        font-weight:900;
        line-height:1;
    }

    .urgency-pill.high{
        background:#fee2e2;
        border-color:#fecaca;
        color:#b91c1c;
    }

    .urgency-pill.medium{
        background:#fef3c7;
        border-color:#fde68a;
        color:#92400e;
    }

    .urgency-pill.low{
        background:#e0f2fe;
        border-color:#bae6fd;
        color:#0369a1;
    }

    .stPlotlyChart{
        border-radius:18px;
        overflow:hidden;
    }

    div[data-testid="stDataFrame"]{
        border-radius:16px;
        overflow:hidden;
    }

    [data-testid="stImage"] img{
        border-radius:16px;
        border:1px solid var(--line);
        background:#fff;
        max-height:360px;
        object-fit:contain;
    }

    div[data-testid="stImage"]{
        text-align:center;
    }

    div[data-testid="stDataFrame"]{
        max-width:100%;
    }

    div[data-testid="stElementContainer"]{
        margin-bottom:.45rem;
    }

    @media (max-width:900px){
        .hero-card{
            align-items:flex-start;
            flex-direction:column;
        }

        .summary-grid,
        .summary-grid.two{
            grid-template-columns:1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">.✦ ݁˖</div>
            <div>
                <div class="sidebar-title">Dashboard</div>
                <div class="sidebar-subtitle">E-Commerce & Logistics</div>
            </div>
        </div>
        <a class="nav-link" href="#dashboard-analytics">Dashboard</a>
        <a class="nav-link" href="#brand-analysis">Sentimen</a>
        <a class="nav-link" href="#top-issue">Isu & Keluhan</a>
        <a class="nav-link" href="#topic-modeling">Topic Modeling</a>
        <a class="nav-link" href="#wordcloud">WordCloud</a>
        <a class="nav-link" href="#social-network-analysis">SNA</a>
        <a class="nav-link" href="#data-tweet">Dataset</a>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("Filter global")
    selected_categories = st.multiselect(
        "Kategori",
        sorted(df["kategori"].dropna().unique()),
        default=sorted(df["kategori"].dropna().unique()),
    )
    selected_brands = st.multiselect(
        "Brand",
        sorted(df["brand"].dropna().unique()),
        default=sorted(df["brand"].dropna().unique()),
    )
    selected_sentiments = st.multiselect(
        "Sentimen",
        ["Negatif", "Netral", "Positif"],
        default=["Negatif", "Netral", "Positif"],
    )
    keyword = st.text_input("Cari tweet / user")

    st.markdown(
        """
        <div class="side-note">
            <strong>E-Commerce & Ekspedisi Analytics</strong>
            Kelompok 5 - Universitas Budi Luhur
        </div>
        """,
        unsafe_allow_html=True,
    )


filtered_df = df[
    df["kategori"].isin(selected_categories)
    & df["brand"].isin(selected_brands)
    & df["Sentiment"].isin(selected_sentiments)
].copy()

if keyword:
    query = keyword.lower()
    mask = (
        filtered_df["full_text"].astype(str).str.lower().str.contains(query, regex=False)
        | filtered_df["username"].astype(str).str.lower().str.contains(query, regex=False)
    )
    filtered_df = filtered_df[mask]

if filtered_df.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini.")
    st.stop()


total_tweets = len(filtered_df)
negative_total = int((filtered_df["Sentiment"] == "Negatif").sum())
negative_share = negative_total / total_tweets * 100 if total_tweets else 0
brand_count = filtered_df["brand"].nunique()
top_issue = issue_df.sort_values("Jumlah", ascending=False).iloc[0]
top_account = top_accounts_df.sort_values("Betweenness", ascending=False).iloc[0]

st.markdown(
    """
    <div id="dashboard-analytics" class="hero-card">
        <div>
            <div class="eyebrow">Dashboard Analytics</div>
            <h1>War Kredibilitas Brand: E-Commerce & Ekspedisi</h1>
            <p>Analisis sentimen, topic modeling, keluhan, dan Social Network Analysis.</p>
        </div>
        <div class="dataset-pill">Twitter / X Dataset</div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)
with metric_cols[0]:
    metric_card("Total Tweet", fmt_int(total_tweets), "Tweet sesuai filter", primary=True)
with metric_cols[1]:
    metric_card("Distribusi Sentimen", fmt_pct(negative_share), "Tweet negatif")
with metric_cols[2]:
    metric_card("Brand Analysis", str(brand_count), "Brand Analisis")
with metric_cols[3]:
    metric_card("Top Issue", fmt_int(top_issue["Jumlah"]), f"{top_issue['Isu']} Terbanyak")
with metric_cols[4]:
    metric_card("Akurasi Model", "82.68%", "XGBoost + TF-IDF")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="medium")
with left.container(border=True):
    section_title("Brand Analysis", "Distribusi Sentimen per Brand", "brand-analysis")
    st.plotly_chart(make_brand_chart(filtered_df), use_container_width=True, config=chart_config())

with right.container(border=True):
    section_title("Komposisi", "Sentimen E-Commerce & Logistics")
    sentiment_fig, sentiment_table = make_sentiment_chart(filtered_df)
    st.plotly_chart(sentiment_fig, use_container_width=True, config=chart_config())
    st.dataframe(sentiment_table, hide_index=True, use_container_width=True, height=140)

left, right = st.columns([1.65, 1], gap="medium")
with left.container(border=True):
    section_title("Top Issue", "Isu Komplain Terbanyak", "top-issue")
    st.plotly_chart(make_issue_chart(filtered_df, issue_df), use_container_width=True, config=chart_config())

with right.container(border=True):
    section_title("Evidence Tweets", "Keluhan User")
    evidence = make_evidence_tweets(filtered_df)
    if evidence.empty:
        st.info("Tidak ada tweet negatif pada filter ini.")
    else:
        evidence_table = evidence.assign(
            User="@" + evidence["username"].astype(str),
            Tweet=evidence["full_text"].map(lambda value: short_text(value, 82)),
        )[["kategori", "User", "Tweet"]]
        st.dataframe(evidence_table, hide_index=True, use_container_width=True, height=270)

left, right = st.columns([1.65, 1], gap="medium")
with left.container(border=True):
    section_title("Topic Modeling", "Klasterisasi Topik", "topic-modeling")
    show_image(IMAGE_DIR / "klasterisasi.png")
    st.markdown(
        f"""
        <div class="eyebrow" style="margin-top:14px">Evaluasi Model Sentimen</div>
        <div class="modeling-title">Model Sentimen XGBoost + TF-IDF</div>
        <div class="summary-grid">
            <div class="summary-box"><span>F1 Macro</span><strong>0.4692</strong></div>
            <div class="summary-box"><span>F1 Weighted</span><strong>0.8081</strong></div>
            <div class="summary-box"><span>Support Test</span><strong>381</strong></div>
        </div>
        {render_model_comparison_table()}
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="topic-panel">
            <div class="score-row">
                <span class="score-label">Silhouette Score</span>
                <strong class="score-pill">0.5003</strong>
            </div>
            <h2>Interpretasi Topik</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for index, topic in enumerate(TOPIC_SUMMARY, start=1):
        cluster_class = f"cluster-{index - 1}"
        tags = "".join(f"<span class='tag {cluster_class}'>{item}</span>" for item in topic["keywords"])
        st.markdown(
            f"""
            <div class="topic-card">
                <span class="topic-count">{topic["count"]}</span>
                <h3>Topik {index}: {topic["title"]}</h3>
                <p>{topic["description"]}</p>
                {tags}
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.container(border=True):
    section_title("WordCloud", "Wordcloud Isu Keluhan", "wordcloud")
    show_image(IMAGE_DIR / "wordcloud.png")

centrality_df = enrich_accounts(top_accounts_df, edges_df)

with st.container(border=True):
    section_title("Social Network Analysis", "Jaringan Akun", "social-network-analysis")
    left, right = st.columns([1.65, 1], gap="medium")
    with left:
        show_image(IMAGE_DIR / "sna_graph.png")

    with right:
        section_title("Centrality", "Akun Prioritas")
        st.markdown(
            f"""
            <div class="summary-grid two">
                <div class="summary-box"><span>Prioritas Utama</span><strong>@{top_account["Akun"]}</strong></div>
                <div class="summary-box"><span>Betweenness</span><strong>{top_account["Betweenness"]:.6f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<span class="search-with-icon"></span>', unsafe_allow_html=True)
        account_query = st.text_input("Cari akun, brand, kategori", key="account_search")

    shown_accounts = centrality_df
    if account_query:
        query = account_query.lower()
        shown_accounts = centrality_df[
            centrality_df.astype(str).apply(
                lambda row: row.str.lower().str.contains(query, regex=False).any(),
                axis=1,
            )
        ]

    section_title("Centrality Table", "Daftar Akun Prioritas")
    st.markdown(
        render_centrality_table(shown_accounts[["Akun", "Brand", "Relasi", "Betweenness", "Urgensi"]].head(12)),
        unsafe_allow_html=True,
    )

with st.container(border=True):
    section_title("Data Tweet", "Tweet Hasil Analisis", "data-tweet")
    table_controls = st.columns([1.2, 1.2, 1])
    with table_controls[0]:
        topic_filter = st.selectbox(
            "Fokus Topik",
            [
                "Data sesuai filter global",
                "Komplain negatif",
                "Paket hilang/rusak/refund",
                "Promo, ongkir, pengiriman",
                "Kurir, resi, status barang",
            ],
        )
    with table_controls[1]:
        table_search = st.text_input("Cari dalam tabel")
    with table_controls[2]:
        row_limit = st.selectbox("Tampil", [25, 50, 100], index=0)

    table_df = filtered_df.copy()
    if topic_filter == "Komplain negatif":
        table_df = table_df[table_df["Sentiment"].eq("Negatif")]
    elif topic_filter != "Data sesuai filter global":
        topic_map = {
            "Paket hilang/rusak/refund": "Paket hilang/rusak/refund",
            "Promo, ongkir, pengiriman": "Promo, ongkir, dan pengiriman",
            "Kurir, resi, status barang": "Kurir, resi, dan status barang",
        }
        table_df = table_df[
            table_df["topic_label"].astype(str).str.contains(topic_map[topic_filter], case=False, na=False)
        ]

    if table_search:
        query = table_search.lower()
        table_df = table_df[
            table_df.astype(str).apply(
                lambda row: row.str.lower().str.contains(query, regex=False).any(),
                axis=1,
            )
        ]

    display_df = table_df.head(row_limit).assign(
        Username=lambda data: "@" + data["username"].astype(str),
        Tweet=lambda data: data["full_text"].map(short_text),
    )
    st.caption(f"{fmt_int(len(table_df))} data sesuai filter")
    st.dataframe(
        display_df[["Username", "brand", "kategori", "Sentiment", "topic_label", "Tweet"]],
        hide_index=True,
        use_container_width=True,
        height=430,
    )

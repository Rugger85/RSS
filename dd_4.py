import os, re, io, html, base64, math, unicodedata, urllib.parse, mimetypes, socket
from io import BytesIO
from datetime import datetime
import hashlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import create_engine, text as _sql_text

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
    TableStyle, NextPageTemplate, PageBreak, Image
)
from reportlab.lib.utils import ImageReader

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    from PyPDF2 import PdfReader, PdfWriter

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Foreign Media Monitoring - DEMP",
    page_icon="https://raw.githubusercontent.com/Rugger85/DEMP-FR/main/logo.jpeg",
    layout="wide"
)
socket.setdefaulttimeout(12.0)
st.sidebar.markdown("""
<style>
/* —— PURE LIGHT FROSTED SIDEBAR (no blue tint) —— */
[data-testid="stSidebar"] {
    position: relative;                 /* enable ::before layer */
    overflow: hidden;                   /* clip the frost layer to sidebar */
}

/* White frost layer that neutralizes color bleed from page background */
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.68), rgba(250,250,250,0.54));
    backdrop-filter: blur(18px) saturate(170%) brightness(1.05);
    -webkit-backdrop-filter: blur(18px) saturate(170%) brightness(1.05);
    border-right: 1px solid rgba(255,255,255,0.45);
    box-shadow: inset 0 0 20px rgba(0,0,0,0.05);
    z-index: 0;                         /* sits behind the content */
}

/* Make the actual sidebar content sit above the frost layer */
[data-testid="stSidebar"] > div {
    position: relative;
    z-index: 1;
}

/* ---- Light Frosted Glass Card in Sidebar ---- */
.sidebar-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.45), rgba(245,245,245,0.25));
    border: 1px solid rgba(255,255,255,0.30);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 14px 0;
    position: relative;
    box-shadow: 0 3px 10px rgba(2,6,23,.15);
}

/* Text styling */
.sidebar-card h4 {
    color: #111;
    font-weight: 800;
    font-size: 1.2rem;
    margin-bottom: 8px;
}
.sidebar-card p {
    color: #333;
    font-weight: 500;
    margin-bottom: 6px;
}
.sidebar-card span {
    color: #ff4d4d;
    font-weight: 700;
}
</style>

<div style="display:flex;gap:10px;margin:14px 0;">
    <div class="sidebar-card">
        <div style="position:absolute;top:12px;right:14px;display:flex;align-items:center;gap:4px;opacity:0.7;"></div>
        <h4>Recent Issues</h4>
        <div style="margin-top:8px;">
            <p>Combine multiple topics into one PDF Report</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)



# Then continue with your sidebar content
#st.sidebar.title("Recent Issues")
#st.sidebar.multiselect("Select topics", ["A", "B", "C"])
# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _norm_topic_val(t: str) -> str:
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip().lower()

def normalize_text(t):
    if not isinstance(t, str):
        return ""
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[^\w\s\-\.,'&:/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

def to_data_uri(local_path: str) -> str | None:
    if not local_path or not os.path.exists(local_path):
        return None
    mime, _ = mimetypes.guess_type(local_path)
    if mime is None:
        mime = "image/jpeg"
    try:
        with open(local_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return None

def _norm_url(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().str.replace(r"/+$", "", regex=True)

# ---- helpers for bulk build/merge ----
def _gather_topic_material(topic: str):
    """Collect header, stats, videos, articles for a topic (no UI)."""
    norm = _norm_topic_val(topic)
    is_local = ("pakistan" in norm) or ("پاکستان" in topic)

    # header from latest result for the topic
    rep_row = (results[results["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)]
               .sort_values("created_at", ascending=False)
               .head(1)
               .to_dict("records"))
    header = rep_row[0] if rep_row else {
        "topic": topic, "ai_insights": "", "ai_summary": "",
        "ai_hashtags": "", "created_at": ""
    }

    # stats + logos already precomputed
    stats = stats_map_all.get(norm, {})

    # videos (same logic you use in detail)
    show_v = total_df_final[total_df_final["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
    if not show_v.empty:
        show_v["__is_english__"] = show_v["title"].apply(is_english_title)
        show_v = show_v[show_v["__is_english__"] == True]
        if is_local:
            show_v = show_v[
                show_v["title"].str.contains(r"\bpakistan\b", case=False, na=False) |
                show_v["title"].str.contains("پاکستان", case=False, na=False)
            ]
        show_v = filter_videos_hard(show_v, topic)
        show_v["published_at"] = pd.to_datetime(show_v["published_at"], errors="coerce")
        show_v["__title_key__"] = show_v["title"].apply(normalize_text)
        show_v = (
            show_v.sort_values(["published_at", "video_id"], ascending=[False, True])
                  .drop_duplicates(subset=["__title_key__", "published_at"], keep="first")
                  .drop(columns=["__title_key__", "__is_english__", "channel_url_norm"], errors="ignore")
        )

    # articles (same logic you use in detail)
    topic_articles = topics[topics["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
    if not topic_articles.empty:
        topic_articles["published"] = pd.to_datetime(topic_articles["published"], errors="coerce")
        topic_articles["__en__"] = topic_articles["title"].fillna("").apply(is_english_title)
        topic_articles = topic_articles[topic_articles["__en__"] == True]
        if is_local:
            topic_articles = topic_articles[
                topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) |
                topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False)
            ]
        else:
            topic_articles = topic_articles[~(
                topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) |
                topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False)
            )]
        topic_articles = topic_articles.sort_values("published", ascending=False)\
                                       .drop_duplicates(subset=["title","published"], keep="first")

    # empty fallbacks to satisfy _pdf_build schema
    if show_v.empty:
        show_v = pd.DataFrame(columns=["title","channel_title","view_count","like_count",
                                       "comment_count","published_at","url","thumbnail",
                                       "channel_url","channel_thumb"])
    if topic_articles.empty:
        topic_articles = pd.DataFrame(columns=["title","source","summary","link","published"])

    return header, stats, show_v, topic_articles


def _merge_pdfs(pdf_bytes_list: list[bytes]) -> bytes:
    """Take a list of PDF bytes and return one merged PDF bytes."""
    writer = PdfWriter()
    for b in pdf_bytes_list:
        try:
            reader = PdfReader(io.BytesIO(b))
            for page in reader.pages:
                writer.add_page(page)
        except Exception:
            pass
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()


def short(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    for div, suf in ((1e9,"B"), (1e6,"M"), (1e3,"k")):
        if abs(n) >= div:
            x = n/div
            return f"{x:.2f}{suf}".rstrip("0").rstrip(".")
    return str(int(n)) if float(n).is_integer() else str(n)

def _placeholder_img(seed: str) -> str:
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
    return f"https://picsum.photos/seed/{h}/800/450"

def _domain_from_url(u: str) -> str:
    try:
        from urllib.parse import urlparse
        d = urlparse(str(u)).netloc
        d = d.split("@")[-1]
        if ":" in d:
            d = d.split(":")[0]
        return d
    except Exception:
        return ""

def _favicon_from_any_url(u: str, size: int = 64) -> str:
    d = _domain_from_url(u)
    if not d:
        return ""
    return f"https://www.google.com/s2/favicons?domain={d}&sz={int(size)}"

def _logo_src_from_row(ch_thumb: str, ch_url: str, fallback_url: str = "") -> str:
    ch_thumb = (ch_thumb or "").strip()
    if ch_thumb:
        return ch_thumb
    src = _favicon_from_any_url(ch_url or fallback_url, 64)
    return src

_THEMES = {
    "UI": {"bg":"#0a0f1f","bg_grad_from":"#0a0f1f","bg_grad_to":"#0e1b33","card":"#0e1629cc",
           "ink":"#6B3F69","muted":"#333333","accent":"#5dd6ff","border":"#1b2740",
           "link":"#6B3F69","desc":"#3e3e3e","card_bg":"#0f1a30","desc_label":"#6B3F69"},
    "PDF": {"ink":"#0e1629","muted":"#334155","accent":"#1d4ed8","desc":"#0ea5e9",
            "border":"#cbd5e1","card":"#f1f5f9","card_alt":"#e2e8f0","demp":"#ff4d4d",
            "band":"#0e1629","band_text":"#ffffff"}
}
THEME = _THEMES["UI"]

# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------
ENGINE = create_engine(
    'postgresql://neondb_owner:npg_2w1oKXamdsOr@ep-divine-lab-a4rip6ll-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
)

@st.cache_data(show_spinner=False, ttl=300)
def load_data():
    videos = pd.read_sql("SELECT * FROM videos;", ENGINE)
    if "title" in videos.columns:
        videos["title"] = videos["title"].apply(normalize_text)

    allow = pd.read_sql("SELECT * FROM channels_allowlist;", ENGINE)
    results = pd.read_sql("SELECT * FROM ai_results;", ENGINE)
    topics = pd.read_sql("SELECT * FROM ai_topics;", ENGINE)
    rss = pd.read_sql("SELECT * FROM rss_feeds;", ENGINE)
    return videos, allow, results, topics, rss

videos, allow, results, topics, rss = load_data()

# Pre-computations for KPIs
_v = videos.copy()
_v["channel_url_norm"] = _norm_url(_v.get("channel_url", ""))
if _v["channel_url_norm"].notna().any() and (_v["channel_url_norm"] != "").any():
    channels_in_videos = _v.loc[_v["channel_url_norm"] != "", "channel_url_norm"].nunique()
elif "channel_id" in _v.columns:
    channels_in_videos = _v["channel_id"].dropna().nunique()
else:
    channels_in_videos = _v["channel_title"].dropna().nunique()

_t = topics.copy()
_r = rss.copy()
unique_articles = _t.dropna(subset=["title"]).assign(t=lambda d: d["title"].apply(normalize_text)).query("t != ''")["t"].nunique()
reports_generated = results.dropna(subset=["topic"]).assign(t=lambda c: c["topic"].apply(_norm_topic_val))["t"].nunique()
unique_video_titles = videos.dropna(subset=["title"]).assign(t=lambda e: e["title"].apply(normalize_text)).query("t != ''")["t"].nunique()
unique_channel_origins = (videos.get("channel_origin", pd.Series(dtype=str)).astype(str).str.strip().replace({"": pd.NA}).dropna().nunique())
unique_web = _r.dropna(subset=["rss_url"]).assign(t=lambda f: f["rss_url"].apply(normalize_text)).query("t != ''")["t"].nunique()

k1, k1_label = f"{reports_generated}", "Reports Generated"
k2, k2_label = f"{channels_in_videos}", "Monitered Channels"
k3, k3_label = f"{unique_web}", "Monitered Web Sources"
k4, k4_label = f"{unique_video_titles}", "Videos Monitered"
k5, k5_label = f"{unique_articles}", "Articles Monitered"
k6, k6_label = f"{unique_channel_origins}", "Countries Covered"
fk1, fk2, fk3, fk4, fk5, fk6 = map(short, (k1, k2, k3, k4, k5, k6))

# --------------------------------------------------------------------------------------
# Relevance logic + helpers used by detail page
# --------------------------------------------------------------------------------------
try:
    from langdetect import detect as _lang_detect
except Exception:
    _lang_detect = None

_ARABIC_URDU_REGEX = re.compile(r'[\u0600-\u06FF]')

def is_english_title(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.strip()
    if _lang_detect is not None:
        try:
            return _lang_detect(t) == "en"
        except Exception:
            pass
    if _ARABIC_URDU_REGEX.search(t):
        return False
    letters = re.findall(r'[A-Za-z]', t)
    ratio = len(letters) / max(1, len(t))
    return ratio >= 0.50

_STOP = {
    "the","a","an","of","and","to","in","on","for","by","with","at","from","as",
    "is","are","was","were","be","been","it","its","this","that","these","those",
    "about","into","over","after","before","under","above","between","within",
    "news","latest","update","report","video","live","full","hd","2025","2024"
}

def _norm_text(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> list[str]:
    return [w for w in _norm_text(s).split() if w and w not in _STOP]

def parse_topic(topic: str):
    t_norm = _norm_text(topic)
    toks = _tokens(topic)
    strong = [w for w in toks if len(w) >= 5]
    return t_norm, toks, strong

def _wb_find(hay: str, needle: str) -> bool:
    if not needle: return False
    return re.search(rf"\b{re.escape(needle)}\b", hay, flags=re.IGNORECASE) is not None

def is_relevant_rule(topic: str, title: str, desc: str = "") -> tuple[bool,str]:
    t_phrase, toks, strong = parse_topic(topic)
    title_n = _norm_text(title)
    desc_n  = _norm_text(desc)

    if t_phrase and t_phrase in title_n:
        return True, "exact_phrase_title"

    hits_title = sum(1 for w in toks if _wb_find(title, w))
    if hits_title >= 1:
        return True, f"tokens_in_title:{hits_title}"

    strong_hits = sum(1 for w in strong if _wb_find(title, w))
    desc_hits   = sum(1 for w in toks if _wb_find(desc, w))
    if strong_hits >= 1 or desc_hits >= 1:
        return True, f"strong_title+desc:{strong_hits}+{desc_hits}"

    hits_desc = sum(1 for w in toks if _wb_find(desc, w))
    if hits_desc >= 2:
        return True, f"tokens_in_desc:{hits_desc}"

    return False, "no_rule_matched"

def filter_videos_hard(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    if df.empty: return df
    work = df.copy()
    if "description" not in work.columns:
        work["description"] = ""
    flags = work.apply(
        lambda r: is_relevant_rule(topic, str(r.get("title","")), str(r.get("description",""))),
        axis=1
    )
    work["__rel_ok__"] = [ok for ok,_ in flags]
    work["__rel_reason__"] = [why for _,why in flags]
    return work[work["__rel_ok__"]].copy()

def is_pk_topic(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return bool(re.search(r"\bpakistan\b", t)) or ("پاکستان" in text)

# Logo & stats maps for detail page
def logos_inline_html(logos: list, max_n: int = 10):
    if not logos:
        return ""
    seen = set()
    items = []
    for thumb, name in logos:
        if not thumb or thumb in seen:
            continue
        seen.add(thumb)
        items.append(
            f'<img src="{html.escape(str(thumb))}" referrerpolicy="no-referrer" '
            f'title="{html.escape(str(name or ""))}" alt="{html.escape(str(name or ""))}" '
            f'style="width:28px;height:28px;border-radius:50%;object-fit:cover;'
            f'border:1px solid rgba(255,255,255,0.25);margin-left:8px">'
        )
        if len(items) >= max_n:
            break
    return "".join(items)

def build_logos_map(df: pd.DataFrame):
    if df.empty:
        return {}
    tmp = df.copy()
    tmp["topic_norm"] = tmp["topic"].apply(_norm_topic_val)
    tmp["channel_url"] = tmp.get("channel_url", "")
    tmp["logo_src"] = tmp.apply(
        lambda r: _logo_src_from_row(str(r.get("channel_thumb","")), str(r.get("channel_url",""))),
        axis=1
    )
    tmp = tmp.dropna(subset=["channel_title"])
    tmp["published_at"] = pd.to_datetime(tmp["published_at"], errors="coerce")
    tmp = tmp.sort_values(["topic_norm", "channel_title", "published_at"], ascending=[True, True, False])\
             .drop_duplicates(subset=["topic_norm", "channel_title"])
    g = tmp.groupby("topic_norm").apply(lambda g: list(zip(g["logo_src"].tolist(), g["channel_title"].tolist())))
    return g.to_dict()

def build_stats_map(df: pd.DataFrame):
    if df.empty:
        return {}
    tmp = df.copy()
    tmp["topic_norm"] = tmp["topic"].apply(_norm_topic_val)
    tmp["published_at"] = pd.to_datetime(tmp["published_at"], errors="coerce")
    tmp["date_only"] = tmp["published_at"].dt.date
    for c in ["view_count", "like_count", "comment_count"]:
        if c not in tmp.columns:
            tmp[c] = 0
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)
    agg = tmp.groupby("topic_norm").agg(
        channels=("channel_title", lambda s: s.dropna().nunique()),
        days=("date_only", lambda s: s.dropna().nunique()),
        views=("view_count", "sum"),
        likes=("like_count", "sum"),
        comments=("comment_count", "sum"),
    ).reset_index()
    out = {}
    for _, r in agg.iterrows():
        out[r["topic_norm"]] = {
            "channels": int(r["channels"] or 0),
            "days": int(r["days"] or 0),
            "views": int(r["views"] or 0),
            "likes": int(r["likes"] or 0),
            "comments": int(r["comments"] or 0),
            "shares": 0
        }
    return out

# Merge once for maps
total_df_final = results.merge(videos, on="topic", how="inner", suffixes=("", "_v"))
logos_map_all = build_logos_map(total_df_final)
stats_map_all = build_stats_map(total_df_final)

def _fmt_num(n: int) -> str:
    try:
        n = int(n)
    except:
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def _demp_percent(stats: dict) -> str:
    v = max(0, int(stats.get("views", 0)))
    l = max(0, int(stats.get("likes", 0)))
    c = max(0, int(stats.get("comments", 0)))
    s = max(0, int(stats.get("shares", 0) or 0))
    score = ((l * 1.2 + c * 1.5 + s * 1.2) / (max(1, v) / 10) * 100.0)
    score = max(0.0, min(score, 99.9))
    return f"{score:.1f}%"

def _clip(txt: str, limit: int) -> str:
    if not isinstance(txt, str):
        return ""
    return txt if len(txt) <= limit else txt[:limit] + "…"

def report_card_html_pro(row: dict, idx: int, logos: list, stats: dict, is_local: bool) -> str:
    topic = row.get("topic", "") or ""
    date_val = row.get("created_at", "")
    date_str = date_val.strftime("%Y-%m-%d %H:%M") if isinstance(date_val, pd.Timestamp) else str(date_val or "")
    hashtags = row.get("ai_hashtags", "") or row.get("hashtags","") or ""
    insights = _clip(row.get("ai_insights", "") or "", 380)
    summary = _clip(row.get("ai_summary", "") or "", 420)
    title_url = "?view=report&topic=" + urllib.parse.quote_plus(topic)
    ch = _fmt_num((stats or {}).get("channels", 0))
    dy = _fmt_num((stats or {}).get("days", 0))
    vw = _fmt_num((stats or {}).get("views", 0))
    lk = _fmt_num((stats or {}).get("likes", 0))
    cm = _fmt_num((stats or {}).get("comments", 0))
    sh = "—"
    demp = _demp_percent(stats or {})
    logos_right = logos_inline_html(logos, max_n=10)
    return f"""
    <div style="display:flex;gap:10px;margin:14px 0;">
    <div style="width:26px;text-align:center;font-weight:700;font-size:16px;color:{THEME['muted']};">{idx}</div>
    <div style="flex:1;background: linear-gradient(180deg, rgba(122, 122, 115,.15), rgba(122, 122, 115,.30));border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:18px;position:relative;box-shadow:0 3px 10px rgba(2,6,23,.25)">
        <div style="position:absolute;top:12px;right:14px;display:flex;align-items:center;">{logos_right}</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <a href="{title_url}" style="color:{THEME['link']};font-weight:1200;font-size:1.25rem;text-decoration:none">{html.escape(topic)}</a>
        </div>
        <div style="color:{THEME['ink']};font-weight:600;margin-bottom:6px;">
        Channels: {ch} • Days: {dy} • Views: {vw} • Likes: {lk} • Comments: {cm} • Shares: {sh} • <span style="color:#ff4d4d;font-weight:800"> Traction Index: {demp}</span>
        </div>
        <div style="color:{THEME['muted']};font-weight:600;margin-bottom:4px;">Date: {html.escape(date_str)} &nbsp;&nbsp; Hashtags: {html.escape(hashtags)}</div>
        <div style="margin-top:8px;">
        <div style="color:{THEME['desc']};font-weight:800;margin-bottom:4px;">AI Insights</div>
        <div style="color:{THEME['ink']};margin-bottom:10px;">{html.escape(insights)}</div>
        <div style="color:{THEME['desc']};font-weight:800;margin-bottom:4px;">Summary</div>
        <div style="color:{THEME['ink']};">{html.escape(summary)}</div>
        </div>
    </div>
    </div>
    """

RENAME_CAMEL = {"video_id":"videoId","channel_id":"channelId","channel_title":"channelTitle","channel_origin":"channelOrigin","channel_thumb":"channelThumb","channel_subscribers":"channelSubscribers","channel_total_views":"channelTotalViews","published_at":"publishedAt","duration_hms":"duration_hms","view_count":"viewCount","like_count":"likeCount","comment_count":"commentCount","privacy_status":"privacyStatus","made_for_kids":"madeForKids","has_captions":"hasCaptions","url":"url","thumbnail":"thumbnail","title":"title","description":"description"}

def _row_to_card_shape(row: dict) -> dict:
    out = dict(row)
    for k, v in list(row.items()):
        if k in RENAME_CAMEL:
            out[RENAME_CAMEL[k]] = v
    for k in ["title","url","thumbnail","channelTitle","channelThumb","channelUrl","description","channelSubscribers","channelTotalViews","channelOrigin","viewCount","likeCount","commentCount","duration_hms","publishedAt","hasCaptions"]:
        out.setdefault(k, "")
    return out

def _fmt_count(v):
    if v is None or v == "":
        return "—"
    try:
        n = int(v)
    except:
        try:
            n = int(float(v))
        except:
            return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def card_markdown_pro(row: dict, idx: int) -> str:
    r = _row_to_card_shape(row)
    title = r.get("title", "")
    url = r.get("url", "")
    thumb = r.get("thumbnail", "")
    ch = r.get("channelTitle", "")
    chlogo = _logo_src_from_row(r.get("channelThumb",""), r.get("channelUrl",""), url)
    subs = _fmt_count(r.get("channelSubscribers"))
    chviews = _fmt_count(r.get("channelTotalViews"))
    country = r.get("channelOrigin", "") or "—"
    views = _fmt_count(r.get("viewCount"))
    likes = _fmt_count(r.get("likeCount"))
    comments = _fmt_count(r.get("commentCount"))
    desc_raw = (r.get("description", "") or "").replace("\n", " ")
    desc = desc_raw[:450] + ("…" if len(desc_raw) > 450 else "")
    dur = r.get("duration_hms") or "—"
    pub = r.get("publishedAt") or "—"
    cap = "No" if not r.get("hasCaptions") else "Yes"
    img_tag = f'<img src="{chlogo}" referrerpolicy="no-referrer" style="width:22px;height:22px;border-radius:50%;object-fit:cover" />' if chlogo else ""
    ch_head = "<div style='display:flex;align-items:center;gap:8px;'>" + f"{img_tag}<span style='font-weight:600;color:#6B3F69;'>{html.escape(str(ch))}</span></div>"
    title_link = f'<a href="{url}" target="_blank" style="text-decoration:none;color:{THEME["link"]};">{html.escape(str(title))}</a>' if url else html.escape(str(title))
    return f"""
<div style="display:flex;gap:8px;margin:10px 0;">
  <div style="width:26px;text-align:center;font-weight:700;font-size:16px;color:{THEME['muted']};">{idx}</div>
  <div style="flex:1;border:1px solid {THEME['border']};border-radius:12px;padding:10px;box-shadow:0 3px 10px rgba(2,6,23,.25);background: linear-gradient(180deg, rgba(122, 122, 115,.15), rgba(122, 122, 115,.30))">
    <div style="font-size:1.08rem;font-weight:700;margin:2px 0 10px 0;color:{THEME['ink']}">{title_link}</div>
    <div style="display:grid;grid-template-columns:230px 1fr 300px;gap:12px;align-items:start;">
      <div>{f'<a href="{url}" target="_blank"><img src="{thumb}" referrerpolicy="no-referrer" style="width:100%;height:auto;border-radius:10px"></a>' if thumb else ''}</div>
      <div>
        <div style="display:flex;gap:24px;font-weight:600;margin-bottom:6px;color:{THEME['ink']}">
          <div>Views: {views}</div><div>Comments: {comments}</div><div>Likes: {likes}</div><div>⏱ {dur} • {pub}</div><div>Captions: {cap}</div>
        </div>
        <div style="color:{THEME['ink']};line-height:1.35;"><span style="font-weight:700;color:{THEME['desc_label']};">Description:</span> {html.escape(desc)}</div>
      </div>
      <div style="border-left:2px dotted #33507a;padding-left:12px">
        {ch_head}
        <div style="margin-top:6px;color:{THEME['ink']}">
          <div>Subscribers: {subs}</div><div>Views: {chviews}</div><div>Country: {html.escape(str(country))}</div>
        </div>
      </div>
    </div>
  </div>
</div>
"""

# --------------------------------------------------------------------------------------
# PDF builder (used by detail view)
# --------------------------------------------------------------------------------------
def _pdf_build(topic, header_row, stats_dict, videos_df, articles_df):
    def _comma(v):
        try:
            n = int(float(v)); return f"{n:,}"
        except Exception:
            return str(v or "0")

    buf = io.BytesIO()
    lm = rm = 18 * mm
    tm = bm = 16 * mm
    doc = BaseDocTemplate(buf, leftMargin=lm, rightMargin=rm, topMargin=tm, bottomMargin=bm, pagesize=A4)
    frame_portrait = Frame(lm, bm, A4[0]-lm-rm, A4[1]-tm-bm, id="portrait")
    L = landscape(A4)
    frame_land = Frame(lm, bm, L[0]-lm-rm, L[1]-tm-bm, id="landscape")

    bg_path = "https://raw.githubusercontent.com/Rugger85/DEMP-FR/main/final_Design_layout_final-01-01.png"

    def _on_portrait(canvas, doc_obj):
        canvas.setPageSize(A4)
        try:
            bg = ImageReader(bg_path)
            canvas.drawImage(bg, 0, 0, width=A4[0], height=A4[1])
        except Exception:
            pass

    def _on_land(canvas, doc_obj):
        canvas.setPageSize(L)

    doc.addPageTemplates([
        PageTemplate(id="Portrait", frames=[frame_portrait], onPage=_on_portrait),
        PageTemplate(id="Landscape", frames=[frame_land], onPage=_on_land)
    ])

    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("h_title", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=22, alignment=1, textColor=colors.HexColor("#01647b"), spaceAfter=10)
    h_topic = ParagraphStyle("h_topic", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, textColor=colors.HexColor("#01647b"), spaceAfter=6)
    label = ParagraphStyle("label", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, textColor=colors.HexColor("#3b4350"), leading=14)
    section = ParagraphStyle("section", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#01647b"), spaceAfter=4, spaceBefore=6)
    tag_style = ParagraphStyle("tags", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, textColor=colors.HexColor("#3b4350"), leading=14)
    table_title = ParagraphStyle("table_title", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, textColor=colors.black, spaceAfter=6)
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=12, textColor=colors.HexColor("#0e1629"), wordWrap="CJK")
    header_style = ParagraphStyle("hdr", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)

    elems = []
    elems.append(Spacer(1, 30 * mm))
    elems.append(Paragraph("Central Monitoring Unit – Digital Media Report", h_title))
    elems.append(Spacer(1, 6 * mm))
    elems.append(Paragraph(f"Topic: {html.escape(topic)}", h_topic))

    created = header_row.get("created_at", "")
    try:
        created_str = created.strftime("%Y-%m-%d %H:%M")
    except Exception:
        created_str = str(created or "")
    elems.append(Paragraph(f"Date: {created_str}", label))

    ch = _comma((stats_dict or {}).get("channels", 0))
    dy = _comma((stats_dict or {}).get("days", 0))
    vw = _comma((stats_dict or {}).get("views", 0))
    lk = _comma((stats_dict or {}).get("likes", 0))
    cm = _comma((stats_dict or {}).get("comments", 0))
    ti = _demp_percent(stats_dict or {})

    elems.append(Paragraph(
        f"Channels: {ch} • Days: {dy} • Views: {vw} • Likes: {lk} • Comments: {cm} • <b>Traction Index: {ti}</b>",
        label
    ))
    elems.append(Spacer(1, 5 * mm))

    ai_insights = html.escape(header_row.get("ai_insights", "") or "")
    summary = html.escape(header_row.get("ai_summary", "") or "")
    hashtags = html.escape(header_row.get("ai_hashtags", "") or "")

    elems.append(Paragraph("AI Insights", section))
    elems.append(Paragraph(ai_insights, label))
    elems.append(Spacer(1, 4 * mm))
    elems.append(Paragraph("Summary", section))
    elems.append(Paragraph(summary, label))
    elems.append(Spacer(1, 4 * mm))
    if hashtags:
        elems.append(Paragraph("Hashtags", section))
        elems.append(Paragraph(hashtags, tag_style))

    elems.append(NextPageTemplate("Landscape"))
    elems.append(PageBreak())

    _PLACEHOLDER_PNG_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAHgAAABQCAYAAABZxZ2mAAAACXBIWXMAAAsSAAALEgHS3X78AAABcElEQVR4nO3aMU7DQBQF4S8"
        "n3Z2lq8l7h7gQyqG1w1t6o3V5y1g0o0s7gY8w0S8yQ3e/0Qq+f3g0G7V8g6h9C4a+qf8F4h2JbCwAAAAAAAAAAAAAA8D9c7R3v1z3x"
        "mRrQeD0q4m8l7bqD3hYV0mJ9G5x1k8s2w3uK2pQy2e6sQ2v8cK4dZr7fKcG9fW2nq6dDkFqS5f2y0W3e5H5nq1m9bq8cJQ0nJ9h0Z/"
        "8n2Jw7ZkZ0b7l3bq6cKk2k8b6u8dJY3r7b0q+qJgQ3j0YHn8pQKAAAAAAAAAAAAAAB8H7S1Q6eFme1AAAAAElFTkSuQmCC"
    )

    def _placeholder_img(max_w, max_h):
        raw = base64.b64decode(_PLACEHOLDER_PNG_B64)
        bio = io.BytesIO(raw)
        img = Image(bio)
        img._restrictSize(max_w, max_h)
        return img

    def _img_from_any(src, max_w, max_h):
        try:
            if not src:
                return _placeholder_img(max_w, max_h)
            if isinstance(src, str) and src.startswith("data:image/"):
                b64 = src.split(",", 1)[1]
                bio = io.BytesIO(base64.b64decode(b64))
                img = Image(bio); img._restrictSize(max_w, max_h); return img
            if isinstance(src, str) and os.path.exists(src):
                img = Image(src); img._restrictSize(max_w, max_h); return img
            if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
                from urllib.request import urlopen, Request
                req = Request(src, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(req, timeout=8) as r:
                    data = r.read()
                bio = io.BytesIO(data)
                img = Image(bio); img._restrictSize(max_w, max_h); return img
        except Exception:
            pass
        return _placeholder_img(max_w, max_h)

    def _favicon_for_url(u, max_w, max_h):
        g = _favicon_from_any_url(u, 64)
        return _img_from_any(g, max_w, max_h)

    def _logo_from_channel(channel_thumb, channel_url, max_w, max_h):
        if isinstance(channel_thumb, str) and channel_thumb.strip():
            img = _img_from_any(channel_thumb, max_w, max_h)
            return img
        try:
            if not isinstance(channel_url, str) or not channel_url:
                return _placeholder_img(max_w, max_h)
            from urllib.request import urlopen, Request
            req = Request(channel_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=8) as r:
                html_text = r.read().decode("utf-8", errors="ignore")
            m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html_text, re.IGNORECASE)
            if not m:
                m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', html_text, re.IGNORECASE)
            if m:
                og_img = m.group(1)
                return _img_from_any(og_img, max_w, max_h)
        except Exception:
            pass
        fav = _favicon_from_any_url(channel_url, 64)
        return _img_from_any(fav, max_w, max_h)

    # Videos table
    elems.append(Paragraph("Relevant Videos", table_title))
    elems.append(Spacer(1, 2 * mm))

    Ls = landscape(A4)
    avail_w = Ls[0] - lm - rm
    ratios = [0.09, 0.07, 0.24, 0.16, 0.07, 0.06, 0.07, 0.12, 0.12]
    col_widths = [r * avail_w for r in ratios]

    rows = [[
        Paragraph("Thumb", header_style),
        Paragraph("Logo", header_style),
        Paragraph("Title", header_style),
        Paragraph("Channel", header_style),
        Paragraph("Views", header_style),
        Paragraph("Likes", header_style),
        Paragraph("Comments", header_style),
        Paragraph("Published", header_style),
        Paragraph("URL", header_style)
    ]]

    vids = videos_df.copy()
    vids["published_at"] = pd.to_datetime(vids.get("published_at"), errors="coerce")
    thumb_box_w, thumb_box_h = col_widths[0], 28
    logo_box_w, logo_box_h = col_widths[1], 24

    for r in vids.to_dict("records"):
        thumb_src = r.get("thumbnail")
        logo_src_thumb = r.get("channel_thumb")
        logo_src_url = r.get("channel_url")
        thumb = _img_from_any(thumb_src, thumb_box_w, thumb_box_h)
        logo = _logo_from_channel(logo_src_thumb, logo_src_url, logo_box_w, logo_box_h)
        rows.append([
            thumb,
            logo,
            Paragraph(html.escape(str(r.get("title", "") or "")), cell),
            Paragraph(html.escape(str(r.get("channel_title", "") or "")), cell),
            _comma(r.get("view_count")),
            _comma(r.get("like_count")),
            _comma(r.get("comment_count")),
            (r["published_at"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["published_at"]) else ""),
            Paragraph(html.escape(str(r.get("url", "") or "")), cell)
        ])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0e1629")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("ALIGN",(4,1),(6,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f1f5f9"),colors.HexColor("#e2e8f0")]),
        ("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#0e1629")),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
        ("BOX",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    elems.append(tbl)

    # Articles table
    elems.append(Spacer(1, 6 * mm))
    elems.append(Paragraph("Relevant Articles", table_title))
    elems.append(Spacer(1, 2 * mm))

    ar = articles_df.copy()
    ar["published"] = pd.to_datetime(ar.get("published"), errors="coerce")

    a_avail_w = Ls[0] - lm - rm
    a_ratios = [0.06, 0.12, 0.48, 0.14, 0.20]
    a_col_w = [r * a_avail_w for r in a_ratios]
    a_cell = ParagraphStyle("a_cell", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=12, textColor=colors.HexColor("#0e1629"), wordWrap="CJK")

    a_rows = [[
        Paragraph("Icon", header_style),
        Paragraph("Source", header_style),
        Paragraph("Title", header_style),
        Paragraph("Published", header_style),
        Paragraph("URL", header_style),
    ]]

    a_icon_w, a_icon_h = a_col_w[0], 18
    for r in ar.to_dict("records"):
        link = r.get("link") or r.get("url") or ""
        src = r.get("source") or r.get("publisher") or ""
        ttl = r.get("title") or ""
        pub = r.get("published")
        pub_s = pub.strftime("%Y-%m-%d %H:%M") if isinstance(pub, pd.Timestamp) and not pd.isna(pub) else (str(pub) if pub else "")
        icon = _favicon_for_url(link, a_icon_w, a_icon_h)
        a_rows.append([
            icon,
            Paragraph(html.escape(str(src)), a_cell),
            Paragraph(html.escape(str(ttl)), a_cell),
            Paragraph(html.escape(pub_s), a_cell),
            Paragraph(html.escape(str(link)), a_cell),
        ])

    a_tbl = Table(a_rows, colWidths=a_col_w, repeatRows=1)
    a_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0e1629")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f1f5f9"),colors.HexColor("#e2e8f0")]),
        ("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#0e1629")),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
        ("BOX",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    elems.append(a_tbl)

    # Channel details table
    elems.append(Spacer(1, 8 * mm))
    elems.append(Paragraph("Channel Details (Videos in this Report)", table_title))
    elems.append(Spacer(1, 2 * mm))

    ch_df = vids.copy()
    if "channel_id" in ch_df.columns and ch_df["channel_id"].notna().any():
        group_key = "channel_id"
    elif "channel_url" in ch_df.columns and ch_df["channel_url"].notna().any():
        group_key = "channel_url"
    else:
        group_key = "channel_title"

    to_num = lambda s: pd.to_numeric(s, errors="coerce").fillna(0)

    has_created = "channel_created_at" in ch_df.columns
    agg_dict = {
        "channel_title": ("channel_title", lambda s: (s.dropna().iloc[0] if not s.dropna().empty else "")),
        "channel_url": ("channel_url", lambda s: (s.dropna().iloc[0] if not s.dropna().empty else "")),
        "channel_thumb": ("channel_thumb", lambda s: (s.dropna().iloc[0] if not s.dropna().empty else "")),
        "channel_origin": ("channel_origin", lambda s: (s.dropna().iloc[0] if not s.dropna().empty else "")),
        "channel_subscribers": ("channel_subscribers", lambda s: int(to_num(s).max())),
        "channel_total_views": ("channel_total_views", lambda s: int(to_num(s).max())),
        "videos_in_report": ("video_id", lambda s: s.dropna().nunique()),
    }
    if has_created:
        agg_dict["channel_created_at"] = ("channel_created_at", lambda s: pd.to_datetime(s, errors="coerce").dropna().min())
    else:
        agg_dict["created_from_videos"] = ("published_at", lambda s: pd.to_datetime(s, errors="coerce").dropna().min())

    ch_agg = ch_df.groupby(group_key, dropna=False).agg(**agg_dict).reset_index(drop=False)

    if has_created:
        ch_agg["created_on"] = ch_agg["channel_created_at"]
    else:
        ch_agg["created_on"] = ch_agg["created_from_videos"]
    ch_agg["created_on"] = pd.to_datetime(ch_agg["created_on"], errors="coerce")

    c_avail_w = Ls[0] - lm - rm
    c_ratios  = [0.08, 0.22, 0.12, 0.14, 0.14, 0.12, 0.09, 0.09]
    c_widths  = [r * c_avail_w for r in c_ratios]
    c_rows = [[
        Paragraph("Logo", header_style),
        Paragraph("Channel", header_style),
        Paragraph("Country", header_style),
        Paragraph("Subscribers", header_style),
        Paragraph("Channel Views", header_style),
        Paragraph("Videos in Report", header_style),
        Paragraph("Created On", header_style),
        Paragraph("URL", header_style),
    ]]

    c_logo_w, c_logo_h = c_widths[0], 22

    for r in ch_agg.to_dict("records"):
        logo = _logo_from_channel(r.get("channel_thumb",""), r.get("channel_url",""), c_logo_w, c_logo_h)
        title = r.get("channel_title","") or ""
        country = r.get("channel_origin","") or ""
        subs = _comma(r.get("channel_subscribers"))
        ch_views = _comma(r.get("channel_total_views"))
        vids_in = int(r.get("videos_in_report") or 0)
        created_on = r.get("created_on")
        created_str = created_on.strftime("%Y-%m-%d") if isinstance(created_on, pd.Timestamp) and not pd.isna(created_on) else ""
        url = r.get("channel_url","") or ""

        c_rows.append([
            logo,
            Paragraph(html.escape(str(title)), cell),
            Paragraph(html.escape(str(country)), cell),
            Paragraph(subs, cell),
            Paragraph(ch_views, cell),
            Paragraph(str(vids_in), cell),
            Paragraph(html.escape(created_str), cell),
            Paragraph(html.escape(url), cell),
        ])

    c_tbl = Table(c_rows, colWidths=c_widths, repeatRows=1)
    c_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0e1629")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f1f5f9"),colors.HexColor("#e2e8f0")]),
        ("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#0e1629")),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
        ("BOX",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    elems.append(c_tbl)

    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()

# --------------------------------------------------------------------------------------
# Detail page view
# --------------------------------------------------------------------------------------
def article_card_markdown(row: dict, idx: int) -> str:
    title = str(row.get("title") or "")
    url = str(row.get("link") or row.get("url") or "")
    src = str(row.get("source") or row.get("publisher") or "").strip() or "Article"
    pub = row.get("published")
    pub_str = ""
    if isinstance(pub, pd.Timestamp):
        pub_str = pub.strftime("%Y-%m-%d %H:%M")
    elif isinstance(pub, str):
        pub_str = pub
    summary = str(row.get("summary") or "").replace("\n", " ")
    summary = summary[:480] + ("…" if len(summary) > 480 else "")
    title_link = f'<a href="{html.escape(url)}" target="_blank" style="text-decoration:none;color:{THEME["link"]};">{html.escape(title)}</a>' if url else html.escape(title)
    icon_src = _favicon_from_any_url(url or src)
    icon_html = f'<img src="{html.escape(icon_src)}" referrerpolicy="no-referrer" style="width:18px;height:18px;border-radius:4px;vertical-align:-3px;margin-right:6px;border:1px solid rgba(255,255,255,.15)"/>' if icon_src else ""
    return f"""
<div style="display:flex;gap:8px;margin:10px 0;">
  <div style="width:26px;text-align:center;font-weight:700;font-size:16px;color:{THEME['muted']};">{idx}</div>
  <div style="flex:1;border:1px solid {THEME['border']};border-radius:12px;padding:12px;box-shadow:0 3px 10px rgba(2,6,23,.25);background: linear-gradient(180deg, rgba(122, 122, 115,.15), rgba(122, 122, 115,.30))">
    <div style="font-size:1.05rem;font-weight:800;margin:2px 0 6px 0;color:{THEME['ink']}">{title_link}</div>
    <div style="display:flex;gap:24px;font-weight:600;margin-bottom:6px;color:{THEME['ink']}">
      <div>Source: {icon_html}{html.escape(src)}</div><div>{html.escape(pub_str)}</div>
    </div>
    <div style="color:{THEME['ink']};line-height:1.35;"><span style="font-weight:700;color:{THEME['desc_label']};">Summary:</span> {html.escape(summary)}</div>
  </div>
</div>
"""

def render_detail_page(topic: str):
    st.markdown("<a href='?' style='text-decoration:none'>&larr; Back to dashboard</a>", unsafe_allow_html=True)
    norm = _norm_topic_val(topic)
    is_local = ("pakistan" in norm) or ("پاکستان" in topic)
    logos = logos_map_all.get(norm, [])
    stats = stats_map_all.get(norm, {})
    rep_row = (results[results["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)]
               .sort_values("created_at", ascending=False).head(1).to_dict("records"))
    header = rep_row[0] if rep_row else {"topic": topic, "ai_insights": "", "ai_summary": "", "ai_hashtags": "", "created_at": ""}

    st.markdown("## AI Reports")
    st.markdown(report_card_html_pro({"topic": topic, **header}, 1, logos, stats, is_local), unsafe_allow_html=True)

    show_v = total_df_final[total_df_final["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
    if not show_v.empty:
        show_v["__is_english__"] = show_v["title"].apply(is_english_title)
        show_v = show_v[show_v["__is_english__"] == True]

        if is_local:
            show_v = show_v[
                show_v["title"].str.contains(r"\bpakistan\b", case=False, na=False) |
                show_v["title"].str.contains("پاکستان", case=False, na=False)
            ]

        show_v = filter_videos_hard(show_v, topic)
        show_v["published_at"] = pd.to_datetime(show_v["published_at"], errors="coerce")
        show_v["__title_key__"] = show_v["title"].apply(normalize_text)
        show_v = (
            show_v.sort_values(["published_at", "video_id"], ascending=[False, True])
                  .drop_duplicates(subset=["__title_key__", "published_at"], keep="first")
                  .drop(columns=["__title_key__", "__is_english__", "channel_url_norm"], errors="ignore")
        )

    st.markdown("### Videos")
    if show_v.empty:
        st.info("No videos match the filters for this topic.")
    else:
        for i, row in enumerate(show_v.to_dict("records"), start=1):
            row["channelThumb"] = row.get("channel_thumb","")
            row["channelUrl"] = row.get("channel_url","")
            st.markdown(card_markdown_pro(row, i), unsafe_allow_html=True)

    topic_articles = topics[topics["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
    if not topic_articles.empty:
        topic_articles["published"] = pd.to_datetime(topic_articles["published"], errors="coerce")
        topic_articles["__en__"] = topic_articles["title"].fillna("").apply(is_english_title)
        topic_articles = topic_articles[topic_articles["__en__"] == True]
        if is_local:
            topic_articles = topic_articles[
                topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) |
                topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False)
            ]
        else:
            topic_articles = topic_articles[~(
                topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) |
                topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False)
            )]
        topic_articles = topic_articles.sort_values("published", ascending=False).drop_duplicates(subset=["title","published"], keep="first")

    st.markdown("### Articles")
    if topic_articles.empty:
        st.info("No articles match the filters for this topic.")
    else:
        for i, row in enumerate(topic_articles.to_dict("records"), start=1):
            st.markdown(article_card_markdown(row, i), unsafe_allow_html=True)

    header["report_logo_url"] = ""
    pdf_bytes = _pdf_build(
        topic,
        header,
        stats,
        show_v if not show_v.empty else pd.DataFrame(columns=["title","channel_title","view_count","like_count","comment_count","published_at","url","thumbnail","channel_url","channel_thumb"]),
        topic_articles if not topic_articles.empty else pd.DataFrame(columns=["title","source","summary","link","published"])
    )
    st.markdown("""
    <style>
    /* Style the Download PDF Report button like the Recent Issues card */
    div[data-testid="stDownloadButton"] > button {
        background: white !important;
        color: #111 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 22px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        transition: all 0.25s ease !important;
    }

    /* Hover lift effect */
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        background: #f8f9fa !important;
    }

    /* Pressed (active) effect */
    div[data-testid="stDownloadButton"] > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    clicked = st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"report_{_norm_topic_val(topic)[:60]}.pdf",
        mime="application/pdf",
        key=f"dl_btn_{_norm_topic_val(topic)}"
    )
    if "reports_downloaded" not in st.session_state:
        st.session_state["reports_downloaded"] = 0
    if clicked:
        st.session_state["reports_downloaded"] += 1

# --------------------------------------------------------------------------------------
# ROUTER – decide detail vs main before drawing the main dashboard
# --------------------------------------------------------------------------------------
def _get_query_param(name: str):
    if hasattr(st, "query_params"):  # Streamlit >= 1.31
        return st.query_params.get(name)
    qp = st.experimental_get_query_params()
    val = qp.get(name)
    if isinstance(val, list):
        return val[0] if val else None
    return val

_view  = _get_query_param("view")
_topic = _get_query_param("topic")
if _topic:
    _topic = urllib.parse.unquote_plus(_topic)

if _view == "report" and _topic:
    render_detail_page(_topic)
    st.stop()

# --------------------------------------------------------------------------------------
# MAIN DASHBOARD (only renders when not in detail view)
# --------------------------------------------------------------------------------------
def render_main():
    # --- HERO ---
    IMG_PATH = r"D:\Downloads\YouTube Project\soft\soft\rainbow-coloured-abstract-low-poly-banner-design\12818.jpg"
    uri = to_data_uri(IMG_PATH)
    if uri is None:
        st.warning("Local image not found or unreadable. Showing an online fallback.")
        uri = "https://images.unsplash.com/photo-1445452916036-9022dfd33aa8?q=80&w=2400&auto=format&fit=crop"

    st.markdown(f"""
    <style>
    .hero-wrap{{
      position: relative;
      height: 75vh;
      border-bottom-left-radius: 40px;
      border-bottom-right-radius: 40px;
      overflow: hidden;
      box-shadow: 0 10px 20px rgba(0,0,0,0.10);
      background-image: url('{uri}');
      background-size: cover;
      background-position: center;
    }}
    .waves-band{{ position:absolute; left:0; right:0; bottom:-1px; height:80px; pointer-events:none; }}
    .waves-band svg{{ display:block; width:100%; height:100%; }}
    .moving-waves use{{ animation: move-forever 15s linear infinite; transform: translate3d(0,0,0); }}
    .moving-waves use:nth-child(1){{ animation-duration:22s; animation-delay:-2s; }}
    .moving-waves use:nth-child(2){{ animation-duration:25s; animation-delay:-3s; }}
    .moving-waves use:nth-child(3){{ animation-duration:28s; animation-delay:-4s; }}
    .moving-waves use:nth-child(4){{ animation-duration:31s; animation-delay:-5s; }}
    .moving-waves use:nth-child(5){{ animation-duration:34s; animation-delay:-6s; }}
    .moving-waves use:nth-child(6){{ animation-duration:37s; animation-delay:-7s; }}
    @keyframes move-forever {{
      0%   {{ transform: translate3d(-90px,0,0); }}
      100% {{ transform: translate3d(85px,0,0); }}
    }}
    </style>

    <div class="hero-wrap">
      <div class="waves-band">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 24 150 40" preserveAspectRatio="none" shape-rendering="auto">
          <defs>
            <path id="gentle-wave" d="M-160 44c30 0 58-18 88-18s58 18 88 18 58-18 88-18 58 18 88 18v44h-352z"/>
          </defs>
          <g class="moving-waves">
            <use href="#gentle-wave" x="48" y="-1" fill="rgba(255,255,255,0.40)" />
            <use href="#gentle-wave" x="48" y="3"  fill="rgba(255,255,255,0.35)" />
            <use href="#gentle-wave" x="48" y="5"  fill="rgba(255,255,255,0.25)" />
            <use href="#gentle-wave" x="48" y="8"  fill="rgba(255,255,255,0.20)" />
            <use href="#gentle-wave" x="48" y="13" fill="rgba(255,255,255,0.15)" />
            <use href="#gentle-wave" x="48" y="16" fill="rgba(255,255,255,0.80)" />
          </g>
        </svg>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- KPI CARDS ---
    st.markdown(f"""
    <style>
    .kpi-wrap {{
      max-width: 1160px;
      margin: -160px auto 60px;
      padding: 0 16px;
    }}
    .kpi-card {{
      position: relative;
      border-radius: 22px;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(255,255,255,.20), rgba(255,255,255,.40));
      backdrop-filter: blur(10px);
      box-shadow: 0 12px 22px rgba(0,0,0,.10);
      z-index: 0;
    }}
    .kpi-card::before {{
      content:"";
      position:absolute;
      left:10%; right:10%; bottom:-28px; height:56px;
      background: rgba(0,0,0,.08);
      border-radius: 30px;
      filter: blur(22px);
      z-index:-1;
    }}

    .kpi-grid {{
      display:grid;
      grid-template-columns:repeat(6, minmax(0,1fr));
      gap: 0;
    }}
    .kpi-cell {{
      padding:34px 28px;
      text-align:center;
      position:relative;
    }}
    .kpi-cell:not(:last-child) {{
      border-right:1px solid rgba(17,24,39,.06);
    }}

    .kpi-num {{
      background: #EBD6FB;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .kpi-title {{
      margin:6px 0 0;
      font-weight:700;
      font-size:16px;
      color:#6B3F69;
    }}

    @media(max-width:1200px){{
      .kpi-grid{{grid-template-columns:repeat(3, minmax(0,1fr));}}
    }}
    @media(max-width:700px){{
      .kpi-grid{{grid-template-columns:1fr 1fr;}}
    }}
    @media(max-width:480px){{
      .kpi-grid{{grid-template-columns:1fr;}}
      .kpi-cell:not(:last-child){{border-right:0;border-bottom:1px solid rgba(17,24,39,.06);}}
    }}
    </style>

    <div class="kpi-wrap">
      <div class="kpi-card">
        <div class="kpi-grid">
          <div class="kpi-cell">
            <h1 class="kpi-num">{fk1}</h1>
            <div class="kpi-title">{html.escape(str("Reports Generated"))}</div>
          </div>
          <div class="kpi-cell">
            <h1 class="kpi-num">{fk2}</h1>
            <div class="kpi-title">{html.escape(str("Monitered Channels"))}</div>
          </div>
          <div class="kpi-cell">
            <h1 class="kpi-num">{fk3}</h1>
            <div class="kpi-title">{html.escape(str("Monitered Web Sources"))}</div>
          </div>
          <div class="kpi-cell">
            <h1 class="kpi-num">{fk4}</h1>
            <div class="kpi-title">{html.escape(str("Videos Monitered"))}</div>
          </div>
          <div class="kpi-cell">
            <h1 class="kpi-num">{fk5}</h1>
            <div class="kpi-title">{html.escape(str("Articles Monitered"))}</div>
          </div>
          <div class="kpi-cell">
            <h1 class="kpi-num">{fk6}</h1>
            <div class="kpi-title">{html.escape(str("Countries Covered"))}</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- CAROUSELS ---
    def build_carousel_items_videos(videos_df: pd.DataFrame, limit: int = 20):
        items = []
        if videos_df is not None and not videos_df.empty:
            vids = videos_df.copy()
            mask = vids["title"].astype(str).str.contains(r"\bPakistan\b", case=False, na=False)
            cols = ["title","thumbnail","url","video_id","published_at","channel_title","channel_thumb","channel_url"]
            cols = [c for c in cols if c in vids.columns]
            vids = vids.loc[mask, cols].copy()

            vids["href"] = vids.get("url", "")
            vids.loc[vids["href"].isna() | (vids["href"] == ""), "href"] = (
                "https://www.youtube.com/watch?v=" + vids["video_id"].astype(str)
            )
            vids["img"] = vids.get("thumbnail", "")
            vids.loc[vids["img"].isna() | (vids["img"] == ""), "img"] = vids["title"].apply(_placeholder_img)

            # publisher + logo
            vids["publisher"] = vids.get("channel_title", "").fillna("")
            vids["logo"] = vids.get("channel_thumb", "")
            # fallback to favicon from channel_url
            if "channel_url" in vids.columns:
                vids.loc[vids["logo"].isna() | (vids["logo"] == ""), "logo"] = vids["channel_url"].apply(_favicon_from_any_url)

            vids["tag"] = "Video"
            if "published_at" in vids.columns:
                vids["__ts"] = pd.to_datetime(vids["published_at"], errors="coerce")
                vids = vids.sort_values("__ts", ascending=False, na_position="last")
            items.extend(dict(href=r["href"], img=r["img"], title=r["title"],
                            tag=r["tag"], publisher=r.get("publisher",""),
                            logo=r.get("logo","")) for _, r in vids.iterrows())

        # de-dupe
        seen, deduped = set(), []
        for it in items:
            if not it["href"] or it["href"] in seen: continue
            seen.add(it["href"]); deduped.append(it)
        return deduped[:limit]


    def build_carousel_items_articles(topics_df: pd.DataFrame, limit: int = 20):
        items = []
        if topics_df is not None and not topics_df.empty:
            arts = topics_df.copy()
            mask = (
                arts["title"].astype(str).str.contains(r"\bPakistan\b", case=False, na=False) |
                arts["summary"].astype(str).str.contains(r"\bPakistan\b", case=False, na=False)
            )

            cols = ["title","link","created_at","thumbnail_url","source"]
            cols = [c for c in cols if c in arts.columns]
            arts = arts.loc[mask, cols].copy()

            arts["href"] = arts.get("link", "")
            arts["img"] = arts.get("thumbnail_url", "")
            arts.loc[arts["img"].isna() | (arts["img"] == ""), "img"] = arts["title"].apply(_placeholder_img)

            arts["publisher"] = arts.get("source", "").fillna("")
            arts["logo"] = arts["href"].apply(_favicon_from_any_url)

            arts["tag"] = "Article"
            if "created_at" in arts.columns:
                arts["__ts"] = pd.to_datetime(arts["created_at"], errors="coerce")
                arts = arts.sort_values("__ts", ascending=False, na_position="last")
            items.extend(dict(href=r.get("href",""), img=r.get("img",""), title=r["title"],
                            tag=r["tag"], publisher=r.get("publisher",""),
                            logo=r.get("logo","")) for _, r in arts.iterrows())

        seen, deduped = set(), []
        for it in items:
            if not it["href"] or it["href"] in seen: continue
            seen.add(it["href"]); deduped.append(it)
        return deduped[:limit]


    items_videos = build_carousel_items_videos(videos, limit=500)
    items_articles = build_carousel_items_articles(topics, limit=500)

    a1, a2 = st.columns([1,1], gap="large")
    with a1:
        st.markdown("<div style='border-right:1px solid rgba(200,200,200,0.3); padding-right:20px;text-align: center;'>"
                    "<h4>Videos on Pakistan</h4></div>", unsafe_allow_html=True)
        slides_html_1 = "\n".join(
            f"""
            <div class="slide">
            <a class="card tilt-card" href="{i['href']}" target="_blank" rel="noopener">
                <img src="{i['img']}" alt="{i['title']}">
                <span class="badge">{i['tag']}</span>
                <div class="caption">
                <div class="title">{i['title']}</div>
                <div class="meta">
                    {f'<img src="{i["logo"]}" alt="" referrerpolicy="no-referrer">' if i.get("logo") else ''}
                    <span class="publisher">{i.get("publisher","")}</span>
                </div>
                </div>
            </a>
            </div>
            """
            for i in items_videos
        )

        components.html(f"""
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tiny-slider@2.9.4/dist/tiny-slider.css">

            <style>
            .carousel-wrap {{
                max-width: 1160px; margin: 16px auto 60px; padding: 0 16px;
            }}
            .tns-outer {{ position: relative; }}
            .tns-ovh   {{ overflow: visible; }}
            .tns-item  {{ padding: 10px; }}

            .card {{
                position: relative; display:block; border-radius: 18px; overflow: hidden;
                box-shadow: 0 12px 24px rgba(0,0,0,.18);
                transform-style: preserve-3d;
                background: #000;
            }}
            .card img {{
                width: 100%; height: 220px; object-fit: cover; display:block;
                transform: translateZ(-8px) scale(1.07);
            }}
            .card .badge {{
                position: absolute; left: 12px; top: 12px;
                background: rgba(124, 58, 237,.92); color:#fff; font-weight:700;
                padding: 6px 10px; border-radius: 9999px; font-size: 12px;
                backdrop-filter: blur(4px);
            }}

            /* Caption area with title + publisher row */
            .card .caption {{
                position:absolute; left:0; right:0; bottom:0;
                padding: 14px 14px 16px;
                color:#fff;
                background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,.60) 70%, rgba(0,0,0,.85) 100%);
            }}
            .card .caption .title {{
                font-weight:700; font-size: 15px; line-height: 1.25; margin-bottom: 8px;
            }}
            .card .caption .meta {{
                display:flex; align-items:center; gap:8px;
                font-weight:600;
            }}
            .card .caption .meta img {{
                width:18px; height:18px; border-radius:50%;
                object-fit:cover; border:1px solid rgba(255,255,255,.25);
            }}
            .card .caption .meta .publisher {{
                font-size: 12px; color: rgba(255,255,255,.9); font-weight:700;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                max-width: 80%;
            }}

            .tilt-card {{ will-change: transform; }}
            .card:hover {{ box-shadow: 0 16px 34px rgba(0,0,0,.25); }}
            </style>

            <div class="carousel-wrap">
            <div class="my-slider">
                {slides_html_1}
            </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/tiny-slider@2.9.4/dist/min/tiny-slider.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/vanilla-tilt@1.8.1/dist/vanilla-tilt.min.js"></script>
            <script>
            var slider = tns({{
                container: '.my-slider',
                items: 3,
                gutter: 12,
                slideBy: 1,
                autoplay: true,
                autoplayTimeout: 2500,
                autoplayButtonOutput: false,
                controls: false,
                nav: false,
                mouseDrag: true,
                loop: true,
                speed: 500,
                responsive: {{
                0:   {{ items: 1 }},
                620: {{ items: 2 }},
                980: {{ items: 3 }},
                1240:{{ items: 4 }}
                }}
            }});

            // Init tilt
            document.querySelectorAll('.tilt-card').forEach(function(el){{
                if (!el.vanillaTilt) {{
                VanillaTilt.init(el, {{
                    max: 10,
                    speed: 400,
                    glare: true,
                    'max-glare': .25
                }});
                }}
            }});
            </script>
            """, height=380, scrolling=False)

    with a2:
        st.markdown("<div style='border-right:1px solid rgba(200,200,200,0.3); padding-right:20px;text-align: center;'>"
                    "<h4>Articles on Pakistan</h4></div>", unsafe_allow_html=True)
        slides_html_2 = "\n".join(
            f"""
            <div class="slide">
            <a class="card tilt-card" href="{i['href']}" target="_blank" rel="noopener">
                <img src="{i['img']}" alt="{i['title']}">
                <span class="badge">{i['tag']}</span>
                <div class="caption">
                <div class="title">{i['title']}</div>
                <div class="meta">
                    {f'<img src="{i["logo"]}" alt="" referrerpolicy="no-referrer">' if i.get("logo") else ''}
                    <span class="publisher">{i.get("publisher","")}</span>
                </div>
                </div>
            </a>
            </div>
            """
            for i in items_articles
        )
        components.html(f"""
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/tiny-slider@2.9.4/dist/tiny-slider.css">

            <style>
            .carousel-wrap {{
                max-width: 1160px; margin: 16px auto 60px; padding: 0 16px;
            }}
            .tns-outer {{ position: relative; }}
            .tns-ovh   {{ overflow: visible; }}
            .tns-item  {{ padding: 10px; }}

            .card {{
                position: relative; display:block; border-radius: 18px; overflow: hidden;
                box-shadow: 0 12px 24px rgba(0,0,0,.18);
                transform-style: preserve-3d;
                background: #000;
            }}
            .card img {{
                width: 100%; height: 220px; object-fit: cover; display:block;
                transform: translateZ(-8px) scale(1.07);
            }}
            .card .badge {{
                position: absolute; left: 12px; top: 12px;
                background: rgba(124, 58, 237,.92); color:#fff; font-weight:700;
                padding: 6px 10px; border-radius: 9999px; font-size: 12px;
                backdrop-filter: blur(4px);
            }}

            /* Caption area with title + publisher row */
            .card .caption {{
                position:absolute; left:0; right:0; bottom:0;
                padding: 14px 14px 16px;
                color:#fff;
                background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,.60) 70%, rgba(0,0,0,.85) 100%);
            }}
            .card .caption .title {{
                font-weight:700; font-size: 15px; line-height: 1.25; margin-bottom: 8px;
            }}
            .card .caption .meta {{
                display:flex; align-items:center; gap:8px;
                font-weight:600;
            }}
            .card .caption .meta img {{
                width:18px; height:18px; border-radius:50%;
                object-fit:cover; border:1px solid rgba(255,255,255,.25);
            }}
            .card .caption .meta .publisher {{
                font-size: 12px; color: rgba(255,255,255,.9); font-weight:700;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                max-width: 80%;
            }}

            .tilt-card {{ will-change: transform; }}
            .card:hover {{ box-shadow: 0 16px 34px rgba(0,0,0,.25); }}
            </style>

            <div class="carousel-wrap">
            <div class="my-slider">
                {slides_html_2}
            </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/tiny-slider@2.9.4/dist/min/tiny-slider.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/vanilla-tilt@1.8.1/dist/vanilla-tilt.min.js"></script>
            <script>
            var slider = tns({{
                container: '.my-slider',
                items: 3,
                gutter: 12,
                slideBy: 1,
                autoplay: true,
                autoplayTimeout: 2500,
                autoplayButtonOutput: false,
                controls: false,
                nav: false,
                mouseDrag: true,
                loop: true,
                speed: 500,
                responsive: {{
                0:   {{ items: 1 }},
                620: {{ items: 2 }},
                980: {{ items: 3 }},
                1240:{{ items: 4 }}
                }}
            }});

            // Init tilt
            document.querySelectorAll('.tilt-card').forEach(function(el){{
                if (!el.vanillaTilt) {{
                VanillaTilt.init(el, {{
                    max: 10,
                    speed: 400,
                    glare: true,
                    'max-glare': .25
                }});
                }}
            }});
            </script>
            """, height=380, scrolling=False)

    # --- AI Reports list (topics -> clickable to detail) ---
    st.markdown("## AI Reports")
    if results.empty:
        st.info("No reports available.")
    else:
        latest_per_topic = results.sort_values("created_at", ascending=False)\
                                  .dropna(subset=["topic"])\
                                  .drop_duplicates(subset=["topic"], keep="first")
        i = 1
        for r in latest_per_topic.to_dict("records"):
            t = r.get("topic", "")
            is_local = is_pk_topic(t)
            norm = _norm_topic_val(t)
            logos = logos_map_all.get(norm, [])
            stats = stats_map_all.get(norm, {})
            st.markdown(report_card_html_pro(r, i, logos, stats, is_local), unsafe_allow_html=True)
            i += 1


with st.sidebar:
    #st.title("Recent Issues")
    #+++++++++++++++#
    # ================================
    # 📎 Multi-topic PDF Merge Helpers
    # ================================

    from io import BytesIO
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception:
        from PyPDF2 import PdfReader, PdfWriter


    def _topic_materials(topic: str):
        """Fetch AI results, stats, videos, and articles for a given topic using same logic as render_detail_page()."""
        norm = _norm_topic_val(topic)
        is_local = ("pakistan" in norm) or ("پاکستان" in topic)

        # --- AI results row (latest)
        rep_row = (
            results[results["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)]
            .sort_values("created_at", ascending=False)
            .head(1)
            .to_dict("records")
        )
        header = rep_row[0] if rep_row else {
            "topic": topic, "ai_insights": "", "ai_summary": "", "ai_hashtags": "", "created_at": ""
        }

        stats = stats_map_all.get(norm, {})

        # --- Videos
        show_v = total_df_final[total_df_final["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
        #show_v = filter_videos_for_topic(show_v, topic, min_rel=0.15)
        show_v = filter_videos_hard(show_v, topic)
        if not show_v.empty:
            show_v["__is_english__"] = show_v["title"].apply(is_english_title)
            show_v = show_v[show_v["__is_english__"] == True]

            if is_local:
                show_v = show_v[
                    show_v["title"].str.contains(r"\bpakistan\b", case=False, na=False) |
                    show_v["title"].str.contains("پاکستان", case=False, na=False)
                ]

            show_v["published_at"] = pd.to_datetime(show_v["published_at"], errors="coerce")
            show_v["__title_key__"] = show_v["title"].apply(normalize_text)
            show_v = (
                show_v.sort_values(["published_at", "video_id"], ascending=[False, True])
                    .drop_duplicates(subset=["__title_key__", "published_at"], keep="first")
                    .drop(columns=["__title_key__", "__is_english__", "channel_url_norm"], errors="ignore")
            )

        # --- Articles
        topic_articles = topics[topics["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
        if not topic_articles.empty:
            topic_articles["published"] = pd.to_datetime(topic_articles["published"], errors="coerce")
            topic_articles["__en__"] = topic_articles["title"].fillna("").apply(is_english_title)
            topic_articles = topic_articles[topic_articles["__en__"] == True]
            if is_local:
                topic_articles = topic_articles[
                    topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) |
                    topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False)
                ]
            else:
                topic_articles = topic_articles[
                    ~(
                        topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) |
                        topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False)
                    )
                ]
            topic_articles = topic_articles.sort_values("published", ascending=False)\
                                        .drop_duplicates(subset=["title", "published"], keep="first")

        vids_df = show_v if not show_v.empty else pd.DataFrame(
            columns=["title","channel_title","view_count","like_count","comment_count","published_at","url","thumbnail","channel_url","channel_thumb"]
        )
        arts_df = topic_articles if not topic_articles.empty else pd.DataFrame(
            columns=["title","source","summary","link","published"]
        )

        return header, stats, vids_df, arts_df


    def build_single_topic_pdf_bytes(topic: str) -> bytes:
        """Build a PDF for one topic using existing _pdf_build()."""
        header, stats, vids_df, arts_df = _topic_materials(topic)
        return _pdf_build(topic, header, stats, vids_df, arts_df)


    def merge_pdfs(pdf_bytes_list: list[bytes]) -> bytes:
        """Merge multiple PDF byte blobs into one combined PDF."""
        writer = PdfWriter()
        for blob in pdf_bytes_list:
            reader = PdfReader(BytesIO(blob))
            for page in reader.pages:
                writer.add_page(page)
        out = BytesIO()
        writer.write(out)
        out.seek(0)
        return out.read()


    # ================================
    # 📎 Streamlit UI: Combine Topics
    # ================================

    st.divider()
    #st.subheader("📎 Combine multiple topics into one PDF")

    _all_topics = (
        results.dropna(subset=["topic"])
            .assign(t=lambda d: d["topic"].apply(_norm_topic_val))
            .sort_values("created_at", ascending=False)
            .drop_duplicates(subset=["t"], keep="first")["topic"]
            .tolist()
    )

    picked_topics = st.multiselect("Select two or more topics to include", options=_all_topics, help="Each topic will retain its own first-page design and AI summary.")
    import streamlit as st

    st.markdown("""
    <style>
    /* Match button with "Recent Issues" card */
    div.stButton > button:first-child {
        background: white;
        color: #111;
        font-weight: 600;
        border: none;
        border-radius: 14px;
        padding: 14px 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.25s ease;
    }

    /* Hover effect for soft elevation */
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        background: #f8f9fa;
    }

    /* Optional: focus/active effect */
    div.stButton > button:first-child:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)



    if st.button("Generate Combined PDF"):
        #⬇️ 
        if not picked_topics:
            st.warning("Please select at least one topic.")
        else:
            blobs = []
            for t in picked_topics:
                try:
                    st.write(f"Generating report for **{t}** ...")
                    blobs.append(build_single_topic_pdf_bytes(t))
                except Exception as e:
                    st.error(f"Error building PDF for '{t}': {e}")
            if blobs:
                merged = merge_pdfs(blobs)
                fname = "combined_" + "_".join(_norm_topic_val(t)[:20] for t in picked_topics)[:80] + ".pdf"
                st.download_button("📥 Download Combined Report", data=merged, file_name=fname, mime="application/pdf", key="dl_combined_pdf")
            else:
                st.info("No reports could be generated.")
    


# Draw main (only if not redirected by router)
render_main()

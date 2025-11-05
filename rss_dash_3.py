import os, re, html, unicodedata, urllib.parse, math, io, base64, socket
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text as _sql_text
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, NextPageTemplate, PageBreak, Image
from reportlab.lib.utils import ImageReader
import plotly.express as px

st.set_page_config(page_title="Foreign Media Monitoring - DEMP", page_icon="https://raw.githubusercontent.com/Rugger85/DEMP-FR/main/logo.jpeg", layout="wide")
socket.setdefaulttimeout(12.0)

THEME = {"bg":"#0a0f1f","bg_grad_from":"#0a0f1f","bg_grad_to":"#0e1b33","card":"#0e1629cc","ink":"#e6edf3","muted":"#9fb3c8","accent":"#5dd6ff","border":"#1b2740","link":"#7dc3ff","desc":"#7ee3ff","card_bg":"#0f1a30","desc_label":"#8fd3ff"}
PDF_COLORS = {"ink":"#0e1629","muted":"#334155","accent":"#1d4ed8","desc":"#0ea5e9","border":"#cbd5e1","card":"#f1f5f9","card_alt":"#e2e8f0","demp":"#ff4d4d","band":"#0e1629","band_text":"#ffffff"}

def normalize_text(t):
    if not isinstance(t, str):
        return ""
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[^\w\s\-\.,'&:/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

def _norm_key(sr: pd.Series) -> pd.Series:
    return sr.astype(str).str.normalize("NFKC").str.replace(r"\s+", " ", regex=True).str.strip().str.lower()

def _norm_topic_val(t: str) -> str:
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip().lower()

def _norm_url(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().str.replace(r"/+$", "", regex=True)

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

def is_pk_topic(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return bool(re.search(r"\bpakistan\b", t)) or ("پاکستان" in text)

def render_title_ticker(rows: pd.DataFrame, title: str, ticker_speed: int = 80, row_gap: int = 12, seamless_scroll: bool = False, height: int = 140):
    if rows.empty:
        st.info(f"No rows for {title}.")
        return
    work = rows.copy()
    for c in ["title", "channel_title", "channel_thumb", "channel_url", "url", "published_at", "video_id"]:
        if c not in work.columns:
            work[c] = ""
    work["published_at"] = pd.to_datetime(work["published_at"], errors="coerce")
    work = work.sort_values("published_at", ascending=False)
    cards = []
    for _, r in work.iterrows():
        vid_title = str(r.get("title", "")).strip()
        ch_name = str(r.get("channel_title", "")).strip()
        ch_logo = _logo_src_from_row(str(r.get("channel_thumb","")), str(r.get("channel_url","")), str(r.get("url","")))
        url = str(r.get("url", "")).strip()
        ts = r.get("published_at")
        latest_str = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else ""
        title_html = (f'<a href="{html.escape(url)}" target="_blank" style="text-decoration:none;color:{THEME["link"]};">{html.escape(vid_title)}</a>') if url else html.escape(vid_title)
        ext_html = (f' <a href="{html.escape(url)}" target="_blank" title="Open link" style="text-decoration:none;color:{THEME["muted"]};">↗</a>') if url else ""
        logo_html = (f'<img src="{html.escape(ch_logo)}" loading="lazy" referrerpolicy="no-referrer" title="{html.escape(ch_name)}" alt="{html.escape(ch_name)}" style="width:20px;height:20px;border-radius:50%;object-fit:cover;margin-right:6px;border:1px solid rgba(255,255,255,0.15)"/>') if ch_logo else ""
        cards.append(f'<div class="card"><div class="col date">{latest_str}</div><div class="col topic">{title_html}{ext_html}</div><div class="col ch">{logo_html}<span class="ch-name">{html.escape(ch_name)}</span></div></div>')
    cards_html = "".join(cards)
    duplicate = (seamless_scroll and len(cards) >= 2)
    inner_html = cards_html + cards_html if duplicate else cards_html
    animate_css = "animation: scroll linear var(--duration) infinite;" if duplicate else ""
    html_str = f"""<!doctype html><html><head><meta charset="utf-8"/><style>
:root{{--bg:{THEME['bg']};--card:{THEME['card']};--ink:{THEME['ink']};--muted:{THEME['muted']};--accent:{THEME['accent']};--gap:{row_gap}px;--duration:60s;}}
body{{margin:0;background:transparent;}}
.wrap{{margin:4px 0 10px 0}}
.title{{color:{THEME['ink']};font-weight:800;margin:0 0 6px 4px;font-size:1.05rem;letter-spacing:.2px}}
.ticker-wrap{{width:100%;overflow:hidden;background:transparent;border-radius:14px;border:1px solid rgba(255,255,255,0.08);backdrop-filter:blur(16px)}}
.ticker{{display:inline-flex;gap:var(--gap);align-items:stretch;padding:8px 10px;{animate_css}will-change:transform}}
@keyframes scroll{{0%{{transform:translateX(0)}}100%{{transform:translateX(-50%)}}}}
.card{{display:grid;grid-template-columns:180px 760px 280px;gap:10px;min-width:1240px;padding:8px 10px;background:var(--card);color:var(--ink);border:1px solid rgba(255,255,255,0.07);border-radius:10px;box-shadow:0 8px 22px rgba(2,6,23,.35)}}
.col{{display:flex;align-items:center;color:var(--ink)}}
.topic{{font-weight:700;font-size:1.02rem}}
.ch .ch-name{{font-weight:600;margin-left:4px}}
.date{{color:{THEME['muted']};font-variant-numeric:tabular-nums}}
.ticker:hover{{animation-play-state:paused}}
</style></head><body>
<div class="wrap">
  <div class="title">{html.escape(title)}</div>
  <div class="ticker-wrap" id="wrap"><div class="ticker" id="ticker">{inner_html}</div></div>
</div>
<script>(function(){{try{{var wrap=document.getElementById('wrap');var ticker=document.getElementById('ticker');function setDuration(){{var wrapW=wrap.clientWidth||1;var tickW=ticker.scrollWidth||wrapW;var secsPerScreen=Math.max(5,{int(ticker_speed)});var duration=secsPerScreen*(0.5*tickW/wrapW);duration=Math.max(duration,10);ticker.style.setProperty('--duration',duration.toFixed(1)+'s');}}setDuration();var to=null;window.addEventListener('resize',function(){{if(to)clearTimeout(to);to=setTimeout(setDuration,150);}});}}catch(e){{}}}})();</script>
</body></html>"""
    st.components.v1.html(html_str, height=height, scrolling=False)

def _clip(txt: str, limit: int) -> str:
    if not isinstance(txt, str):
        return ""
    return txt if len(txt) <= limit else txt[:limit] + "…"

def build_logos_map(df: pd.DataFrame):
    if df.empty:
        return {}
    tmp = df.copy()
    tmp["topic_norm"] = tmp["topic"].apply(_norm_topic_val)
    tmp["channel_url"] = tmp.get("channel_url", "")
    tmp["logo_src"] = tmp.apply(lambda r: _logo_src_from_row(str(r.get("channel_thumb","")), str(r.get("channel_url",""))), axis=1)
    tmp = tmp.dropna(subset=["channel_title"])
    tmp["published_at"] = pd.to_datetime(tmp["published_at"], errors="coerce")
    tmp = tmp.sort_values(["topic_norm", "channel_title", "published_at"], ascending=[True, True, False]).drop_duplicates(subset=["topic_norm", "channel_title"])
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
        out[r["topic_norm"]] = {"channels": int(r["channels"] or 0),"days": int(r["days"] or 0),"views": int(r["views"] or 0),"likes": int(r["likes"] or 0),"comments": int(r["comments"] or 0),"shares": 0}
    return out

def logos_inline_html(logos: list, max_n: int = 10):
    if not logos:
        return ""
    seen = set()
    items = []
    for thumb, name in logos:
        if not thumb or thumb in seen:
            continue
        seen.add(thumb)
        items.append(f'<img src="{html.escape(str(thumb))}" referrerpolicy="no-referrer" title="{html.escape(str(name or ""))}" alt="{html.escape(str(name or ""))}" style="width:28px;height:28px;border-radius:50%;object-fit:cover;border:1px solid rgba(255,255,255,0.25);margin-left:8px">')
        if len(items) >= max_n:
            break
    return "".join(items)

def _demp_percent(stats: dict) -> str:
    v = max(0, int(stats.get("views", 0)))
    l = max(0, int(stats.get("likes", 0)))
    c = max(0, int(stats.get("comments", 0)))
    s = max(0, int(stats.get("shares", 0) or 0))
    score = ((l * 1.2 + c * 1.5 + s * 1.2) / (max(1, v) / 10) * 100.0)
    score = max(0.0, min(score, 99.9))
    return f"{score:.1f}%"

RENAME_CAMEL = {"video_id":"videoId","channel_id":"channelId","channel_title":"channelTitle","channel_origin":"channelOrigin","channel_thumb":"channelThumb","channel_subscribers":"channelSubscribers","channel_total_views":"channelTotalViews","published_at":"publishedAt","duration_hms":"duration_hms","view_count":"viewCount","like_count":"likeCount","comment_count":"commentCount","privacy_status":"privacyStatus","made_for_kids":"madeForKids","has_captions":"hasCaptions","url":"url","thumbnail":"thumbnail","title":"title","description":"description"}

def _row_to_card_shape(row: dict) -> dict:
    out = dict(row)
    for k, v in list(row.items()):
        if k in RENAME_CAMEL:
            out[RENAME_CAMEL[k]] = v
    for k in ["title","url","thumbnail","channelTitle","channelThumb","channelUrl","description","channelSubscribers","channelTotalViews","channelOrigin","viewCount","likeCount","commentCount","duration_hms","publishedAt","hasCaptions"]:
        out.setdefault(k, "")
    return out

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
    ch_head = "<div style='display:flex;align-items:center;gap:8px;'>" + f"{img_tag}<span style='font-weight:600;color:#ffffff;'>{html.escape(str(ch))}</span></div>"
    title_link = f'<a href="{url}" target="_blank" style="text-decoration:none;color:{THEME["link"]};">{html.escape(str(title))}</a>' if url else html.escape(str(title))
    return f"""
<div style="display:flex;gap:8px;margin:10px 0;">
  <div style="width:26px;text-align:center;font-weight:700;font-size:16px;color:{THEME['muted']};">{idx}</div>
  <div style="flex:1;border:1px solid {THEME['border']};border-radius:12px;padding:10px;box-shadow:0 3px 10px rgba(2,6,23,.25);background:{THEME['card']}">
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
  <div style="flex:1;border:1px solid {THEME['border']};border-radius:12px;padding:12px;box-shadow:0 3px 10px rgba(2,6,23,.25);background:{THEME['card']}">
    <div style="font-size:1.05rem;font-weight:800;margin:2px 0 6px 0;color:{THEME['ink']}">{title_link}</div>
    <div style="display:flex;gap:24px;font-weight:600;margin-bottom:6px;color:{THEME['ink']}">
      <div>Source: {icon_html}{html.escape(src)}</div><div>{html.escape(pub_str)}</div>
    </div>
    <div style="color:{THEME['ink']};line-height:1.35;"><span style="font-weight:700;color:{THEME['desc_label']};">Summary:</span> {html.escape(summary)}</div>
  </div>
</div>
"""

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

engine = create_engine('postgresql://neondb_owner:npg_2w1oKXamdsOr@ep-solitary-leaf-a4e9snsy-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require')

videos = pd.read_sql("SELECT * FROM videos;", engine)
if "title" in videos.columns:
    videos["title"] = videos["title"].apply(normalize_text)
allow = pd.read_sql("SELECT * FROM channels_allowlist;", engine)
results = pd.read_sql("SELECT * FROM ai_results;", engine)
topics_df = pd.read_sql("SELECT * FROM ai_topics;", engine)

if "published" in topics_df.columns:
    topics_df["published"] = pd.to_datetime(topics_df["published"], errors="coerce", utc=True).dt.tz_convert(None)
for c in ["title","source","summary","link","topic"]:
    if c not in topics_df.columns:
        topics_df[c] = ""

total_df_final = results.merge(videos, on="topic", how="inner", suffixes=("", "_v"))
if "created_at" in total_df_final.columns:
    total_df_final["created_at"] = pd.to_datetime(total_df_final["created_at"], errors="coerce")
if "published_at" in total_df_final.columns:
    total_df_final["published_at"] = pd.to_datetime(total_df_final["published_at"], errors="coerce")

results_local = results[results["topic"].str.contains("Pakistan", case=False, na=False)]
results_int = results[~results["topic"].str.contains("Pakistan", case=False, na=False)].copy()

videos_local = videos.copy()
videos_local = videos_local[videos_local["title"].str.contains(r"\bpakistan\b", case=False, na=False) | videos_local.get("title","").astype(str).str.contains("پاکستان", case=False, na=False)]
videos_local["topic"] = videos_local.get("topic")
videos_int = videos.copy()
videos_int["channel_url_norm"] = _norm_url(videos_int.get("channel_url", ""))
allow["channel_url_norm"] = _norm_url(allow.get("channel_url", ""))
#videos_int = videos_int[videos_int["channel_url_norm"].isin(allow["channel_url_norm"])]
if "channel_origin" in videos_int.columns:
    videos_int = videos_int[videos_int["channel_origin"] != "Pakistan"]

articles_local = topics_df[topics_df["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) | topics_df["title"].astype(str).str.contains("پاکستان", case=False, na=False)].copy()
articles_int = topics_df[~(topics_df["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) | topics_df["title"].astype(str).str.contains("پاکستان", case=False, na=False))].copy()

logos_map_all = build_logos_map(total_df_final)
stats_map_all = build_stats_map(total_df_final)

_videos_channels = videos.copy()
_videos_channels["channel_url_norm"] = _norm_url(_videos_channels.get("channel_url", ""))
if "published_at" in _videos_channels.columns:
    _videos_channels["published_at"] = pd.to_datetime(_videos_channels["published_at"], errors="coerce")
    _videos_channels = _videos_channels.sort_values("published_at", ascending=False).drop_duplicates(subset=["channel_url_norm"], keep="first")
else:
    _videos_channels = _videos_channels.drop_duplicates(subset=["channel_url_norm"], keep="first")

allow["channel_url_norm"] = _norm_url(allow.get("channel_url", ""))
_not_allowed = _videos_channels[~_videos_channels["channel_url_norm"].isin(allow["channel_url_norm"])]
_not_allowed["__pick_label__"] = _not_allowed.apply(lambda r: f"{str(r.get('channel_title','') or '').strip()}", axis=1)

params = st.query_params
view = (params.get("view") or "").strip().lower()
topic_q = params.get("topic")

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
  <div style="flex:1;background:{THEME['card']};border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:18px;position:relative;box-shadow:0 3px 10px rgba(2,6,23,.25)">
    <div style="position:absolute;top:12px;right:14px;display:flex;align-items:center;">{logos_right}</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <a href="{title_url}" style="color:{THEME['link']};font-weight:800;font-size:1.05rem;text-decoration:none">{html.escape(topic)}</a>
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
    doc.addPageTemplates([PageTemplate(id="Portrait", frames=[frame_portrait], onPage=_on_portrait),PageTemplate(id="Landscape", frames=[frame_land], onPage=_on_land)])
    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("h_title", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=22, alignment=1, textColor=colors.HexColor("#01647b"), spaceAfter=10)
    h_topic = ParagraphStyle("h_topic", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, textColor=colors.HexColor("#01647b"), spaceAfter=6)
    label = ParagraphStyle("label", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, textColor=colors.HexColor("#3b4350"), leading=14)
    section = ParagraphStyle("section", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#01647b"), spaceAfter=4, spaceBefore=6)
    tag_style = ParagraphStyle("tags", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5, textColor=colors.HexColor("#3b4350"), leading=14)
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
        f"Channels: {ch} • Days: {dy} • Views: {vw} • Likes: {lk} • Comments: {cm} • "
        f"<b>Traction Index: {ti}</b>", 
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
    table_title = ParagraphStyle("table_title", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, textColor=colors.black, spaceAfter=6)
    elems.append(Paragraph("Relevant Videos", table_title))
    elems.append(Spacer(1, 2 * mm))
    avail_w = L[0] - lm - rm
    ratios = [0.09, 0.07, 0.24, 0.16, 0.07, 0.06, 0.07, 0.12, 0.12]
    col_widths = [r * avail_w for r in ratios]
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=12, textColor=colors.HexColor("#0e1629"), wordWrap="CJK")
    header_style = ParagraphStyle("hdr", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, textColor=colors.white)
    rows = [[Paragraph("Thumb", header_style), Paragraph("Logo", header_style), Paragraph("Title", header_style),Paragraph("Channel", header_style), Paragraph("Views", header_style), Paragraph("Likes", header_style),Paragraph("Comments", header_style), Paragraph("Published", header_style), Paragraph("URL", header_style)]]
    _PLACEHOLDER_PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAHgAAABQCAYAAABZxZ2mAAAACXBIWXMAAAsSAAALEgHS3X78AAABcElEQVR4nO3aMU7DQBQF4S8"
        "n3Z2lq8l7h7gQyqG1w1t6o3V5y1g0o0s7gY8w0S8yQ3e/0Qq+f3g0G7V8g6h9C4a+qf8F4h2JbCwAAAAAAAAAAAAAA8D9c7R3v1z3x"
        "mRrQeD0q4m8l7bqD3hYV0mJ9G5x1k8s2w3uK2pQy2e6sQ2v8cK4dZr7fKcG9fW2nq6dDkFqS5f2y0W3e5H5nq1m9bq8cJQ0nJ9h0Z/"
        "8n2Jw7ZkZ0b7l3bq6cKk2k8b6u8dJY3r7b0q+qJgQ3j0YHn8pQKAAAAAAAAAAAAAAB8H7S1Q6eFme1AAAAAElFTkSuQmCC")
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
    def _favicon_for_url(u, max_w, max_h):
        g = _favicon_from_any_url(u, 64)
        return _img_from_any(g, max_w, max_h)
    thumb_box_w, thumb_box_h = col_widths[0], 28
    logo_box_w, logo_box_h = col_widths[1], 24
    vids = videos_df.copy()
    vids["published_at"] = pd.to_datetime(vids.get("published_at"), errors="coerce")
    for r in vids.to_dict("records"):
        thumb_src = r.get("thumbnail")
        logo_src_thumb = r.get("channel_thumb")
        logo_src_url = r.get("channel_url")
        thumb = _img_from_any(thumb_src, thumb_box_w, thumb_box_h)
        logo = _logo_from_channel(logo_src_thumb, logo_src_url, logo_box_w, logo_box_h)
        rows.append([thumb, logo, Paragraph(html.escape(str(r.get("title", "") or "")), cell), Paragraph(html.escape(str(r.get("channel_title", "") or "")), cell), _comma(r.get("view_count")), _comma(r.get("like_count")), _comma(r.get("comment_count")), (r["published_at"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["published_at"]) else ""), Paragraph(html.escape(str(r.get("url", "") or "")), cell)])
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0e1629")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),8),("ALIGN",(4,1),(6,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f1f5f9"),colors.HexColor("#e2e8f0")]),("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#0e1629")),("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),("BOX",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    elems.append(tbl)
    elems.append(Spacer(1, 6 * mm))
    elems.append(Paragraph("Relevant Articles", table_title))
    elems.append(Spacer(1, 2 * mm))
    ar = articles_df.copy()
    st.dataframe(ar)
    ar["published"] = pd.to_datetime(ar.get("published"), errors="coerce")
    a_avail_w = L[0] - lm - rm
    a_ratios = [0.06, 0.12, 0.48, 0.14, 0.20]
    a_col_w = [r * a_avail_w for r in a_ratios]
    a_cell = ParagraphStyle("a_cell", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=12, textColor=colors.HexColor("#0e1629"), wordWrap="CJK")
    a_header = [Paragraph("Icon", header_style), Paragraph("Source", header_style), Paragraph("Title", header_style), Paragraph("Published", header_style), Paragraph("URL", header_style)]
    a_rows = [a_header]
    a_icon_w, a_icon_h = a_col_w[0], 18
    for r in ar.to_dict("records"):
        link = r.get("link") or r.get("url") or ""
        src = r.get("source") or r.get("publisher") or ""
        ttl = r.get("title") or ""
        pub = r.get("published")
        pub_s = pub.strftime("%Y-%m-%d %H:%M") if isinstance(pub, pd.Timestamp) and not pd.isna(pub) else (str(pub) if pub else "")
        icon = _favicon_for_url(link, a_icon_w, a_icon_h)
        a_rows.append([icon, Paragraph(html.escape(str(src)), a_cell), Paragraph(html.escape(str(ttl)), a_cell), Paragraph(html.escape(pub_s), a_cell), Paragraph(html.escape(str(link)), a_cell)])
    a_tbl = Table(a_rows, colWidths=a_col_w, repeatRows=1)
    a_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0e1629")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f1f5f9"),colors.HexColor("#e2e8f0")]),("TEXTCOLOR",(0,1),(-1,-1),colors.HexColor("#0e1629")),("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),("BOX",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    elems.append(a_tbl)
    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()

def render_detail_page(topic: str):
    st.markdown("<a href='?' style='text-decoration:none'>&larr; Back to dashboard</a>", unsafe_allow_html=True)
    norm = _norm_topic_val(topic)
    is_local = ("pakistan" in norm) or ("پاکستان" in topic)
    logos = logos_map_all.get(norm, [])
    stats = stats_map_all.get(norm, {})
    rep_row = (results[results["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].sort_values("created_at", ascending=False).head(1).to_dict("records"))
    header = rep_row[0] if rep_row else {"topic": topic, "ai_insights": "", "ai_summary": "", "ai_hashtags": "", "created_at": ""}
    st.markdown("## AI Reports")
    st.markdown(report_card_html_pro({"topic": topic, **header}, 1, logos, stats, is_local), unsafe_allow_html=True)
    show_v = total_df_final[total_df_final["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
    if not show_v.empty:
        show_v["__is_english__"] = show_v["title"].apply(is_english_title)
        show_v = show_v[show_v["__is_english__"] == True]
        if is_local:
            show_v = show_v[show_v["title"].str.contains(r"\bpakistan\b", case=False, na=False) | show_v["title"].str.contains("پاکستان", case=False, na=False)]
        else:
            #show_v["channel_url_norm"] = _norm_url(show_v.get("channel_url", ""))
            #allow_set = set(allow["channel_url_norm"].tolist())
            #show_v = show_v[show_v["channel_url_norm"].isin(allow_set)]
        show_v["published_at"] = pd.to_datetime(show_v["published_at"], errors="coerce")
        show_v["__title_key__"] = show_v["title"].apply(normalize_text)
        show_v = (show_v.sort_values(["published_at", "video_id"], ascending=[False, True]).drop_duplicates(subset=["__title_key__", "published_at"], keep="first").drop(columns=["__title_key__", "__is_english__", "channel_url_norm"], errors="ignore"))
    st.markdown("### Videos")
    if show_v.empty:
        st.info("No videos match the filters for this topic.")
    else:
        for i, row in enumerate(show_v.to_dict("records"), start=1):
            row["channelThumb"] = row.get("channel_thumb","")
            row["channelUrl"] = row.get("channel_url","")
            st.markdown(card_markdown_pro(row, i), unsafe_allow_html=True)
    topic_articles = topics_df[topics_df["topic"].apply(lambda x: _norm_topic_val(str(x)) == norm)].copy()
    if not topic_articles.empty:
        topic_articles["published"] = pd.to_datetime(topic_articles["published"], errors="coerce")
        topic_articles["__en__"] = topic_articles["title"].fillna("").apply(is_english_title)
        topic_articles = topic_articles[topic_articles["__en__"] == True]
        if is_local:
            topic_articles = topic_articles[topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) | topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False)]
        else:
            topic_articles = topic_articles[~(topic_articles["title"].astype(str).str.contains(r"\bpakistan\b", case=False, na=False) | topic_articles["title"].astype(str).str.contains("پاکستان", case=False, na=False))]
        topic_articles = topic_articles.sort_values("published", ascending=False).drop_duplicates(subset=["title","published"], keep="first")
    st.markdown("### Articles")
    if topic_articles.empty:
        st.info("No articles match the filters for this topic.")
    else:
        for i, row in enumerate(topic_articles.to_dict("records"), start=1):
            st.markdown(article_card_markdown(row, i), unsafe_allow_html=True)
    header["report_logo_url"] = ""
    pdf_bytes = _pdf_build(topic, header, stats, show_v if not show_v.empty else pd.DataFrame(columns=["title","channel_title","view_count","like_count","comment_count","published_at","url","thumbnail","channel_url","channel_thumb"]), topic_articles if not topic_articles.empty else pd.DataFrame(columns=["title","source","summary","link","published"]))
    clicked = st.download_button(label="⬇️ Download PDF Report", data=pdf_bytes, file_name=f"report_{_norm_topic_val(topic)[:60]}.pdf", mime="application/pdf", key=f"dl_btn_{_norm_topic_val(topic)}")
    if "reports_downloaded" not in st.session_state:
        st.session_state["reports_downloaded"] = 0
    if clicked:
        st.session_state["reports_downloaded"] += 1

def _kpi_card_html(title: str, value: str) -> str:
    return f"""
<div style="background:{THEME['card']};border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:14px 16px;box-shadow:0 3px 10px rgba(2,6,23,.18);">
  <div style="color:{THEME['muted']};font-weight:700;margin-bottom:6px">{html.escape(title)}</div>
  <div style="color:{THEME['ink']};font-weight:900;font-size:1.6rem;letter-spacing:.4px">{html.escape(str(value))}</div>
</div>
"""

params = st.query_params
if (params.get("view") or "").strip().lower() == "report" and params.get("topic"):
    render_detail_page(params.get("topic"))
else:
    if "reports_downloaded" not in st.session_state:
        st.session_state["reports_downloaded"] = 0
    _v = videos.copy()
    _v["channel_url_norm"] = _norm_url(_v.get("channel_url", ""))
    if _v["channel_url_norm"].notna().any() and (_v["channel_url_norm"] != "").any():
        channels_in_videos = _v.loc[_v["channel_url_norm"] != "", "channel_url_norm"].nunique()
    elif "channel_id" in _v.columns:
        channels_in_videos = _v["channel_id"].dropna().nunique()
    else:
        channels_in_videos = _v["channel_title"].dropna().nunique()
    _t = topics_df.copy()
    unique_articles = _t.dropna(subset=["title"]).assign(t=lambda d: d["title"].apply(normalize_text)).query("t != ''")["t"].nunique()
    reports_generated = results.dropna(subset=["topic"]).assign(t=lambda d: d["topic"].apply(_norm_topic_val))["t"].nunique()
    unique_video_titles = videos.dropna(subset=["title"]).assign(t=lambda d: d["title"].apply(normalize_text)).query("t != ''")["t"].nunique()
    unique_channel_origins = (videos.get("channel_origin", pd.Series(dtype=str)).astype(str).str.strip().replace({"": pd.NA}).dropna().nunique())
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        st.markdown(_kpi_card_html("Monitored Channels", _fmt_num(channels_in_videos)), unsafe_allow_html=True)
    with b2:
        st.markdown(_kpi_card_html("Reports Generated", _fmt_num(reports_generated)), unsafe_allow_html=True)
    with b3:
        st.markdown(_kpi_card_html("Videos Monitered", _fmt_num(unique_video_titles)), unsafe_allow_html=True)
    with b4:
        st.markdown(_kpi_card_html("Countries", _fmt_num(unique_channel_origins)), unsafe_allow_html=True)
    with b5:
        st.markdown(_kpi_card_html("Articles", _fmt_num(unique_articles)), unsafe_allow_html=True)
    st.title("Recent Issues")
    with st.sidebar:
        ticker_speed = st.slider("Ticker speed (seconds per screen)", 10, 120, 80, 1)
        row_gap = st.slider("Card gap (px)", 8, 48, 12, 1)
        seamless = st.checkbox("Seamless scroll (duplicate content)", value=True)
        st.caption("Local ticker uses title+description filter containing 'Pakistan'. International ticker uses allow-list & non-Pakistan origin for videos; articles exclude Pakistan keywords.")
        st.divider()
        st.subheader("Allow-list updater")
        options = _not_allowed["__pick_label__"].tolist()
        picked = st.multiselect("Add channels (not currently in allow-list)", options=options, help="Select channels to append into channels_allowlist")
        if picked:
            st.caption(f"Selected: {len(picked)} channel(s)")
        if st.button("➕ Append to allow-list"):
            if not picked:
                st.warning("Select one or more channels first.")
            else:
                rows_to_add = _not_allowed[_not_allowed["__pick_label__"].isin(picked)].copy()
                allow_schema_df = pd.read_sql("SELECT * FROM channels_allowlist LIMIT 0;", engine)
                db_cols = [c for c in allow_schema_df.columns if c.lower() != "id"]
                data_for_insert = {}
                for col in db_cols:
                    if col in rows_to_add.columns:
                        data_for_insert[col] = rows_to_add[col]
                    else:
                        if col == "channel_url" and "channel_url" in rows_to_add.columns:
                            data_for_insert[col] = rows_to_add["channel_url"]
                        elif col == "channel_title" and "channel_title" in rows_to_add.columns:
                            data_for_insert[col] = rows_to_add["channel_title"]
                        elif col == "channel_id" and "channel_id" in rows_to_add.columns:
                            data_for_insert[col] = rows_to_add["channel_id"]
                        elif col == "channel_thumb" and "channel_thumb" in rows_to_add.columns:
                            data_for_insert[col] = rows_to_add["channel_thumb"]
                        elif col == "country" and "channel_origin" in rows_to_add.columns:
                            data_for_insert[col] = rows_to_add["channel_origin"]
                        else:
                            data_for_insert[col] = pd.NA
                to_insert_df = pd.DataFrame(data_for_insert)
                if "channel_url" in to_insert_df.columns:
                    to_insert_df = to_insert_df.dropna(subset=["channel_url"]).drop_duplicates(subset=["channel_url"])
                if not to_insert_df.empty:
                    to_insert_df[db_cols].to_sql("channels_allowlist", con=engine, if_exists="append", index=False)
                    st.success(f"Appended {len(to_insert_df)} channel(s) to allow-list.")
                    allow = pd.read_sql("SELECT * FROM channels_allowlist;", engine)
                    allow["channel_url_norm"] = _norm_url(allow.get("channel_url", ""))
                    _not_allowed_mask = ~_videos_channels["channel_url_norm"].isin(allow["channel_url_norm"])
                    _not_allowed = _videos_channels[_not_allowed_mask].copy()
                    _not_allowed["__pick_label__"] = _not_allowed.apply(lambda r: f"{str(r.get('channel_title','') or '').strip()} — {str(r.get('channel_url','') or '').strip()}", axis=1)
                else:
                    st.info("Nothing to insert (duplicates or empty selection).")
        st.divider()
        st.subheader("Remove from allow-list")
        allow_options = allow.apply(lambda r: f"{r.get('channel_title','') or ''} — {r.get('channel_url','') or ''}", axis=1).tolist()
        remove_choice = st.selectbox("Select channel to remove", options=[""] + allow_options, index=0, help="Pick a channel from the current allow-list to delete")
        if st.button("🗑 Remove from allow-list"):
            if not remove_choice or remove_choice.strip() == "":
                st.warning("Please select a channel first.")
            else:
                url_part = remove_choice.split("—")[-1].strip()
                try:
                    with engine.begin() as conn:
                        conn.execute(_sql_text("DELETE FROM channels_allowlist WHERE channel_url = :url"), {"url": url_part})
                    st.success(f"Removed: {remove_choice}")
                    allow = pd.read_sql("SELECT * FROM channels_allowlist;", engine)
                    allow["channel_url_norm"] = _norm_url(allow.get("channel_url", ""))
                    _not_allowed_mask = ~_videos_channels["channel_url_norm"].isin(allow["channel_url_norm"])
                    _not_allowed = _videos_channels[_not_allowed_mask].copy()
                    _not_allowed["__pick_label__"] = _not_allowed.apply(lambda r: f"{str(r.get('channel_title','') or '').strip()} — {str(r.get('channel_url','') or '').strip()}", axis=1)
                except Exception as e:
                    st.error(f"Error removing channel: {e}")
    a1, a2 = st.columns([7, 5])
    with a1:
        ticker_rows_local_v = videos_local.copy()
        for c in ["title","channel_title","channel_thumb","channel_url","url","published_at","video_id"]:
            if c not in ticker_rows_local_v.columns:
                ticker_rows_local_v[c] = ""
        ticker_rows_local_v["__en__"] = ticker_rows_local_v["title"].fillna("").apply(is_english_title)
        ticker_rows_local_v = ticker_rows_local_v[ticker_rows_local_v["__en__"] == True]
        ticker_rows_local_v["__title_key__"] = ticker_rows_local_v["title"].fillna("").apply(normalize_text)
        ticker_rows_local_v["published_at"] = pd.to_datetime(ticker_rows_local_v["published_at"], errors="coerce")
        ticker_rows_local_v = (ticker_rows_local_v.sort_values(["published_at","video_id"], ascending=[False, True]).drop_duplicates(subset=["__title_key__","published_at"], keep="first").drop(columns=["__en__","__title_key__"], errors="ignore"))
        ticker_rows_local_a = pd.DataFrame({
            "title": topics_df.loc[articles_local.index, "title"].astype(str),
            "channel_title": topics_df.loc[articles_local.index, "source"].astype(str).fillna("Article"),
            "channel_thumb": "",
            "channel_url": topics_df.loc[articles_local.index, "link"].astype(str),
            "url": topics_df.loc[articles_local.index, "link"].astype(str),
            "published_at": topics_df.loc[articles_local.index, "published"],
            "video_id": ""
        })
        ticker_rows_local = pd.concat([ticker_rows_local_v[["title","channel_title","channel_thumb","channel_url","url","published_at","video_id"]], ticker_rows_local_a], ignore_index=True)
        st.markdown("Pakistan's Issues")
        render_title_ticker(ticker_rows_local, title="", ticker_speed=max(6, int(ticker_speed * 0.8)), row_gap=max(6, int(row_gap * 0.6)), seamless_scroll=seamless, height=100)
        ticker_rows_int_v = videos_int.copy()
        for c in ["title","channel_title","channel_thumb","channel_url","url","published_at","video_id"]:
            if c not in ticker_rows_int_v.columns:
                ticker_rows_int_v[c] = ""
        ticker_rows_int_v["__en__"] = ticker_rows_int_v["title"].fillna("").apply(is_english_title)
        ticker_rows_int_v = ticker_rows_int_v[ticker_rows_int_v["__en__"] == True]
        ticker_rows_int_v["__title_key__"] = ticker_rows_int_v["title"].fillna("").apply(normalize_text)
        ticker_rows_int_v["published_at"] = pd.to_datetime(ticker_rows_int_v["published_at"], errors="coerce")
        ticker_rows_int_v = (ticker_rows_int_v.sort_values(["published_at","video_id"], ascending=[False, True]).drop_duplicates(subset=["__title_key__","published_at"], keep="first").drop(columns=["__en__","__title_key__"], errors="ignore"))
        ticker_rows_int_a = pd.DataFrame({
            "title": topics_df.loc[articles_int.index, "title"].astype(str),
            "channel_title": topics_df.loc[articles_int.index, "source"].astype(str).fillna("Article"),
            "channel_thumb": "",
            "channel_url": topics_df.loc[articles_int.index, "link"].astype(str),
            "url": topics_df.loc[articles_int.index, "link"].astype(str),
            "published_at": topics_df.loc[articles_int.index, "published"],
            "video_id": ""
        })
        ticker_rows_int = pd.concat([ticker_rows_int_v[["title","channel_title","channel_thumb","channel_url","url","published_at","video_id"]], ticker_rows_int_a], ignore_index=True)
        st.markdown("International")
        render_title_ticker(ticker_rows_int, title="", ticker_speed=max(6, int(ticker_speed * 0.8)), row_gap=max(6, int(row_gap * 0.6)), seamless_scroll=seamless, height=100)
    with a2:
    import pycountry

    # 1️⃣ Clean and standardize the origin codes (like US, IN, PK)
    origins = (
        videos.get("channel_origin", pd.Series(dtype=str))
              .astype(str)
              .str.strip()
              .str.upper()
              .replace({"": pd.NA, "NAN": pd.NA})
              .dropna()
    )

    # 2️⃣ Convert ISO-2 or name to ISO-3 (for Plotly)
    def to_iso3(x: str):
        aliases = {"UK": "GBR", "UAE": "ARE", "KSA": "SAU", "USA": "USA"}
        if x in aliases:
            return aliases[x]
        c = pycountry.countries.get(alpha_2=x)
        if c:
            return c.alpha_3
        try:
            return pycountry.countries.lookup(x).alpha_3
        except Exception:
            return None

    iso3 = origins.map(to_iso3)
    country_counts = (
        iso3.dropna()
            .value_counts()
            .rename_axis("iso3")
            .reset_index(name="Videos")
    )

    # 3️⃣ Plotly choropleth with ISO-3 codes
    if not country_counts.empty:
        import plotly.express as px
        fig = px.choropleth(
            country_counts,
            locations="iso3",
            locationmode="ISO-3",
            color="Videos",
            color_continuous_scale="Blues",
            scope="world",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font_color=THEME["ink"],
            coloraxis_showscale=False,
            height=420
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(f"""
<div style="background:{THEME['card']};border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:14px 16px;box-shadow:0 3px 10px rgba(2,6,23,.18);color:{THEME['muted']};font-weight:700;">
  Channel Origins Map — No countries found
</div>
""", unsafe_allow_html=True)


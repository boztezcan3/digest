# digest.py — run: python digest.py
import json, os, re, html, hashlib, difflib, time, datetime as dt
import feedparser, requests

CFG = json.load(open("config.json"))
STATE_PATH, OUT = "state.json", "index.html"
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("MODEL", "google/gemini-2.5-flash-lite")
API = "https://openrouter.ai/api/v1/chat/completions"

def norm(t):  return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def entry_dt(e):
    for k in ("published_parsed", "updated_parsed"):
        if e.get(k): return dt.datetime.fromtimestamp(time.mktime(e[k]), dt.timezone.utc)
    return None

def clean(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()

def fetch(cutoff):
    out = []
    for src in CFG["sources"]:
        try:
            feed = feedparser.parse(src["url"], agent="personal-digest/1.0")
        except Exception as ex:
            print("feed failed:", src["url"], ex); continue
        for e in feed.entries:
            ts, link = entry_dt(e), e.get("link", "")
            if (ts and ts < cutoff) or not link: continue
            title = clean(e.get("title", ""))
            title = re.sub(r"\s+-\s+[^-]{2,40}$", "", title)   # strip " - AJC.com"
            if not title: continue
            out.append({
                "id": hashlib.md5(norm(link).encode()).hexdigest()[:12],
                "title": title, "link": link, "source": src["name"],
                "section": src["section"], "ts": ts.isoformat() if ts else "",
                "blurb": clean(e.get("summary", ""))[:500],
            })
    return out

def dedupe(items):
    kept = []
    for it in items:
        if any(difflib.SequenceMatcher(None, norm(it["title"]), norm(k["title"])).ratio() > 0.85
               for k in kept):
            continue
        kept.append(it)
    return kept

PROMPT = """You are editing a personal news digest for one reader.

Sections: us-politics, global, turkey, atlanta.
Reader's interests: US politics, Turkish domestic politics, Atlanta/Georgia, major global events.
Turkish sources disagree sharply along pro-government vs. opposition lines - flag framing in the summary when it matters.
For 'global', keep only genuinely significant international stories with cross-source weight; drop routine items.

For each item you consider worth the reader's attention (score >= 5), return:
  i, score (1-10 relevance to this reader), section, summary, open (true for at most ONE item per section)

Rule for `summary`: at most 25 words, one or two sentences. Use ONLY facts present in the title or blurb. Never invent names, numbers, or outcomes. If the blurb is empty, describe only what the headline states. No preamble, no "this article".

Return JSON only: {"picks":[{"i":3,"score":9,"section":"turkey","summary":"...","open":true}]}"""

def ask(items):
    payload = [{"i": n, "source": it["source"], "section": it["section"],
                "title": it["title"], "blurb": it["blurb"]} for n, it in enumerate(items)]
    r = requests.post(API, timeout=180,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "temperature": 0.2,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": PROMPT},
                           {"role": "user", "content": json.dumps(payload)}]})
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(txt)["picks"]
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0))["picks"] if m else []

def assemble(items, picks):
    final = []
    for p in picks:
        if not isinstance(p.get("i"), int) or p["i"] >= len(items): continue
        it = items[p["i"]]
        if p.get("section") not in CFG["quotas"]: p["section"] = it["section"]
        final.append({**it, "score": p.get("score", 5),
                      "section": p["section"], "summary": p.get("summary", ""),
                      "open": bool(p.get("open"))})
    chosen = []
    for sec, quota in CFG["quotas"].items():
        group = sorted([f for f in final if f["section"] == sec],
                       key=lambda x: (-x["open"], -x["score"]))
        chosen += group[:quota]
    return chosen

def render(chosen):
    by_sec = {}
    for it in chosen: by_sec.setdefault(it["section"], []).append(it)
    when = dt.datetime.now(dt.timezone.utc).strftime("%A, %B %-d %Y")
    parts = [f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>Digest {when}</title>
<style>
 body{{max-width:44rem;margin:2rem auto;padding:0 1rem;font:16px/1.5 -apple-system,system-ui,sans-serif;color:#111}}
 h1{{font-size:1.4rem;margin:0 0 .25rem}} .sub{{color:#666;font-size:.85rem;margin-bottom:2rem}}
 h2{{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:#666;border-bottom:1px solid #ddd;
     padding-bottom:.3rem;margin:2rem 0 .75rem}}
 li{{margin:0 0 1rem 0;list-style:none}} a{{color:#0b4f9e;text-decoration:none}} a:hover{{text-decoration:underline}}
 .s{{display:block;color:#333;margin-top:.15rem}} .m{{color:#888;font-size:.75rem;margin-top:.2rem}}
 .pick a{{font-weight:600}} .pick .star{{color:#c9820a;margin-right:.25rem}}
 ul{{padding:0}}
</style><h1>Digest</h1><div class="sub">{when} &middot; {len(chosen)} stories &middot; &#9733; = open this one</div>"""]
    for sec, quota in CFG["quotas"].items():
        group = by_sec.get(sec, [])
        if not group: continue
        parts.append(f"<h2>{sec.replace('-', ' ')}</h2><ul>")
        for it in group:
            pick = " class=pick" if it["open"] else ""
            star = "<span class=star>&#9733;</span>" if it["open"] else ""
            date = it["ts"][:10] if it["ts"] else ""
            parts.append(f'<li{pick}>{star}<a href="{it["link"]}" target="_blank" rel="noopener">{it["title"]}</a>'
                         f'<span class=s>{it["summary"]}</span>'
                         f'<span class=m>{it["source"]} &middot; {date}</span></li>')
        parts.append("</ul>")
    parts.append("</html>")
    open(OUT, "w").write("\n".join(parts))

def main():
    state = json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {"seen": []}
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=CFG["window_days"])
    items = dedupe(fetch(cutoff))

    seen = set(state["seen"])
    fresh = [it for it in items if it["id"] not in seen]
    print(f"{len(items)} items, {len(fresh)} new")
    if not fresh:
        render([]); return

    picks = ask(fresh)
    chosen = assemble(fresh, picks)
    render(chosen)

    state["seen"] = (state["seen"] + [it["id"] for it in fresh])[-800:]
    json.dump(state, open(STATE_PATH, "w"), indent=1)

if __name__ == "__main__":
    main()

# digest.py
import json, os, re, glob, html, hashlib, difflib, time, datetime as dt
import feedparser, requests

CFG = json.load(open("config.json"))
STATE_PATH, OUT, ARCHIVE = "state.json", "index.html", "archive"
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("MODEL", "google/gemini-2.5-flash-lite")
API = "https://openrouter.ai/api/v1/chat/completions"

SECTION_LABELS = {
    "us-politics": "US Politics", "global": "Global",
    "turkey": "Turkey", "atlanta": "Atlanta",
}
SKIP_URL_PATTERNS = ("/video/", "/podcasts/", "/interactive/")

def norm(t): return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def entry_dt(e):
    for k in ("published_parsed", "updated_parsed"):
        if e.get(k): return dt.datetime.fromtimestamp(time.mktime(e[k]), dt.timezone.utc)
    return None

def clean(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()

def strip_suffix(title):
    # "Headline - The Atlanta Journal-Constitution" -> "Headline"
    if " - " in title:
        head, tail = title.rsplit(" - ", 1)
        if 0 < len(tail) <= 45:
            return head.strip()
    return title

def fetch(cutoff):
    out = []
    default_cap = CFG.get("max_per_source", 25)
    for src in CFG["sources"]:
        cap = src.get("cap", default_cap)
        try:
            feed = feedparser.parse(src["url"], agent="personal-digest/1.0")
        except Exception as ex:
            print("feed failed:", src["name"], ex); continue
        got = []
        for e in feed.entries:
            ts, link = entry_dt(e), e.get("link", "")
            if (ts and ts < cutoff) or not link: continue
            if any(p in link for p in SKIP_URL_PATTERNS): continue
            title = strip_suffix(clean(e.get("title", "")))
            if not title: continue
            got.append({
                "id": hashlib.md5(norm(link).encode()).hexdigest()[:12],
                "title": title, "link": link, "source": src["name"],
                "section": src["section"], "ts": ts.isoformat() if ts else "",
                "blurb": clean(e.get("summary", ""))[:500],
            })
        got.sort(key=lambda x: x["ts"], reverse=True)
        if not got:
            print(f"  ! {src['name']}: 0 items in window")
        out += got[:cap]
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

LANGUAGE: several Turkey sources publish in Turkish (BBC Turkce, Cumhuriyet, Sozcu, T24, Now TV).
Write every summary in English, translating faithfully. Keep proper nouns in their normal English form.

SOURCES: multiple sources often cover the same event. Prefer a spread of distinct events over
several angles on one story - the selection stage enforces source diversity, so score an item on
how much it adds, not on how well known its outlet is.
Turkish sources disagree sharply along pro-government vs. opposition lines - flag framing in the
summary when it matters.
For 'global', keep only genuinely significant international stories with cross-source weight.

For each item you consider worth the reader's attention (score >= 5), return:
  i, score (1-10 relevance to this reader), section, summary, open (true for at most ONE item per section)

Rule for `summary`: at most 25 words, one or two sentences. Use ONLY facts present in the title or
blurb. Never invent names, numbers, or outcomes. If the blurb is empty, describe only what the
headline states. No preamble, no "this article".

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

def choose_for_section(group, quota):
    """One starred item, then round-robin across sources so no single outlet fills the section."""
    group = sorted(group, key=lambda x: (-x["score"], x["ts"]))
    out = [it for it in group if it["open"]][:1]
    placed = {id(x) for x in out}

    buckets = {}
    for it in group:
        if id(it) in placed: continue
        buckets.setdefault(it["source"], []).append(it)

    order = sorted(buckets, key=lambda s: -buckets[s][0]["score"])
    while len(out) < quota:
        moved = False
        for s in order:
            if len(out) >= quota: break
            if buckets[s]:
                out.append(buckets[s].pop(0)); moved = True
        if not moved: break
    return out

def assemble(items, picks):
    final = []
    for p in picks:
        if not isinstance(p.get("i"), int) or p["i"] >= len(items): continue
        it = items[p["i"]]
        sec = str(p.get("section", "")).strip().lower()
        if sec not in CFG["quotas"]: sec = it["section"]
        final.append({**it, "score": p.get("score", 5), "section": sec,
                      "summary": p.get("summary", ""), "open": bool(p.get("open"))})

    chosen = []
    for sec, quota in CFG["quotas"].items():
        chosen += choose_for_section([f for f in final if f["section"] == sec], quota)
    return chosen

def render_html(chosen, home_href, arch_href):
    by_sec = {}
    for it in chosen: by_sec.setdefault(it["section"], []).append(it)
    when = dt.datetime.now(dt.timezone.utc).strftime("%A, %B %-d %Y")

    head = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>Digest {when}</title>
<style>
 body{{max-width:44rem;margin:2rem auto;padding:0 1rem;font:16px/1.5 -apple-system,system-ui,sans-serif;color:#111}}
 h1{{font-size:1.4rem;margin:0 0 .25rem}} .sub{{color:#666;font-size:.85rem}}
 nav{{font-size:.78rem;margin:.5rem 0 2rem}} nav a{{color:#0b4f9e}}
 h2{{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:#666;border-bottom:1px solid #ddd;
     padding-bottom:.3rem;margin:2rem 0 .75rem}}
 ul{{list-style:none;padding:0}} li{{margin:0 0 1rem 0}}
 a{{color:#0b4f9e;text-decoration:none}} a:hover{{text-decoration:underline}}
 .s{{display:block;color:#333;margin-top:.15rem}} .m{{color:#888;font-size:.75rem;margin-top:.2rem}}
 .pick a{{font-weight:600}} .pick .star{{color:#c9820a;margin-right:.25rem}}
</style></head><body>
<h1>Digest</h1>
<div class="sub">{when} &middot; {len(chosen)} stories &middot; &#9733; = open this one</div>
<nav><a href="{arch_href}">Past digests</a></nav>"""

    body = []
    for sec, quota in CFG["quotas"].items():
        group = by_sec.get(sec, [])
        if not group: continue
        body.append(f"<h2>{SECTION_LABELS.get(sec, sec.title())} <small>({len(group)})</small></h2><ul>")
        for it in group:
            pick = " class=pick" if it["open"] else ""
            star = "<span class=star>&#9733;</span>" if it["open"] else ""
            date = it["ts"][:10] if it["ts"] else ""
            body.append(f'<li{pick}>{star}<a href="{it["link"]}" target="_blank" rel="noopener">'
                        f'{it["title"]}</a><span class=s>{it["summary"]}</span>'
                        f'<span class=m>{it["source"]} &middot; {date}</span></li>')
        body.append("</ul>")
    if not chosen:
        body.append("<p>No stories matched in this window.</p>")
    return head + "\n".join(body) + "\n</body></html>"

def write_archive_index():
    os.makedirs(ARCHIVE, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ARCHIVE, "digest-*.html")), reverse=True)
    items = "\n".join(
        f'<li><a href="{os.path.basename(f)}">{os.path.basename(f)[7:-5]}</a></li>'
        for f in files) or "<li>No archived digests yet.</li>"
    open(os.path.join(ARCHIVE, "index.html"), "w").write(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Past digests</title>
<style>
 body{{max-width:32rem;margin:2rem auto;padding:0 1rem;
      font:16px/1.6 -apple-system,system-ui,sans-serif;color:#111}}
 h1{{font-size:1.3rem}} ul{{list-style:none;padding:0}}
 li{{margin:0 0 .5rem}} a{{color:#0b4f9e;text-decoration:none}} a:hover{{text-decoration:underline}}
</style></head><body>
<h1>Past digests</h1>
<nav><a href="../index.html">&larr; Current digest</a></nav>
<ul>
{items}
</ul></body></html>""")

def publish(chosen):
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    os.makedirs(ARCHIVE, exist_ok=True)
    open(OUT, "w").write(render_html(chosen, "", "archive/index.html"))
    if chosen:
        open(os.path.join(ARCHIVE, f"digest-{day}.html"), "w").write(
            render_html(chosen, "../index.html", "../archive/index.html"))
    write_archive_index()

def main():
    state = json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {"seen": []}
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=CFG["window_days"])
    items = dedupe(fetch(cutoff))

    seen = set(state["seen"])
    fresh = [it for it in items if it["id"] not in seen]
    print(f"{len(items)} items after caps/dedupe, {len(fresh)} new")

    if not fresh:
        print("No new items since the last run - keeping the previous digest.")
        if not os.path.exists(OUT):
            publish([])
        return

    picks = ask(fresh)
    chosen = assemble(fresh, picks)
    counts = {}
    for it in chosen: counts[it["source"]] = counts.get(it["source"], 0) + 1
    print("selected by source:", counts)

    publish(chosen)

    state["seen"] = (state["seen"] + [it["id"] for it in fresh])[-800:]
    json.dump(state, open(STATE_PATH, "w"), indent=1)

if __name__ == "__main__":
    main()

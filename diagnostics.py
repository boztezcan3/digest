"""
diagnostics.py - per-run diagnostics for the digest, written to files.

Files written (all under ./diagnostics/ unless DIGEST_DIAG_DIR is set):

  latest.txt               human-readable report of the most recent run
  runs.jsonl               one compact summary line per run (for trends)
  runs/run-<stamp>.json    full detail for each run (last KEEP_RUN_FILES kept)

Every public function is wrapped so that an error here can never break
the digest: it prints one warning line and carries on.

Typical wiring (see the notes that came with this file):
    diag.start_run(model)             at the start of the program
    diag.feed(...) / feed_error(...)  inside fetch()
    diag.usage(label, result)         inside call_llm()
    diag.selection(...)               after each per-section selection call
    diag.section_result(...)          inside assemble()'s quota loop
    diag.clusters(report)             inside assign_event_ids()
    diag.published(chosen)            at the end of assemble()
    diag.finish_run(status, error)    last thing, even after a crash
"""

import datetime as dt
import functools
import json
import os
import statistics
import time
from collections import Counter

DIAG_DIR = os.environ.get("DIGEST_DIAG_DIR", "diagnostics")
RUNS_DIR = os.path.join(DIAG_DIR, "runs")
JSONL_PATH = os.path.join(DIAG_DIR, "runs.jsonl")
REPORT_PATH = os.path.join(DIAG_DIR, "latest.txt")

KEEP_RUN_FILES = 60          # full per-run JSON files to keep
CROSS_DAY_DAYS = 3           # how far back to look for repeated stories
CROSS_DAY_THRESHOLD = 0.30   # lexical similarity that flags a possible repeat
POSITION_BIAS_SHARE = 0.60   # picks in first 2/5 of the list that triggers a hint
MAX_EVENTS = 60

RUN = {}


# ----------------------------------------------------------------- plumbing

def _now():
    return dt.datetime.now(dt.timezone.utc)


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # diagnostics must never break the digest
            print(f"[diagnostics] {fn.__name__} failed: "
                  f"{type(exc).__name__}: {exc}")
            return None
    return wrapper


def _blank(model=""):
    return {
        "started": _now().isoformat(),
        "model": model,
        "status": "running",
        "error": None,
        "notes": {},
        "counters": {},
        "events": [],
        "feeds": [],
        "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                  "cost": 0.0, "by_label": {}},
        "selection": {},
        "sections": {},
        "clusters": None,
        "repairs": {},
        "published": [],
        "repeats": [],
    }


def _run():
    if "started" not in RUN:
        RUN.update(_blank())
        RUN["_t0"] = time.time()
    return RUN


@_safe
def start_run(model=""):
    RUN.clear()
    RUN.update(_blank(model))
    RUN["_t0"] = time.time()


# ------------------------------------------------------------ simple events

@_safe
def note(key, value):
    _run()["notes"][key] = value


@_safe
def count(name, n=1):
    counters = _run()["counters"]
    counters[name] = counters.get(name, 0) + n


@_safe
def event(kind, detail=""):
    events = _run()["events"]
    if len(events) < MAX_EVENTS:
        events.append({"kind": kind, "detail": str(detail)[:300]})


@_safe
def repair(types):
    """types: dict like {'omitted_candidate': 2, 'mixed_news_opinion': 1}"""
    _run()["repairs"] = dict(types or {})


# --------------------------------------------------------------------- feeds

@_safe
def feed(section, name, returned, in_window, unseen, sent):
    _run()["feeds"].append({
        "section": section, "source": name, "returned": returned,
        "in_window": in_window, "unseen": unseen, "sent": sent,
        "error": None,
    })


@_safe
def feed_error(section, name, error):
    _run()["feeds"].append({
        "section": section, "source": name, "returned": 0, "in_window": 0,
        "unseen": 0, "sent": 0, "error": str(error)[:200],
    })


# --------------------------------------------------------------------- usage

@_safe
def usage(label, result):
    """result: the parsed JSON body of an OpenRouter/OpenAI-style reply."""
    u = (result or {}).get("usage") or {}
    prompt = int(u.get("prompt_tokens") or 0)
    completion = int(u.get("completion_tokens") or 0)
    cost = u.get("cost")

    total = _run()["usage"]
    total["calls"] += 1
    total["prompt_tokens"] += prompt
    total["completion_tokens"] += completion
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        total["cost"] += cost

    by = total["by_label"].setdefault(
        label or "unlabeled",
        {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
    )
    by["calls"] += 1
    by["prompt_tokens"] += prompt
    by["completion_tokens"] += completion


# ----------------------------------------------------------------- selection

@_safe
def selection(section, items, picks):
    """
    items: the candidate list sent for this section (in the order sent).
    picks: the model's valid picks, with i = index WITHIN items.
    """
    n = len(items)
    picked = sorted({p["i"] for p in picks})

    quintiles = [0, 0, 0, 0, 0]
    for i in picked:
        quintiles[min(4, int(i / n * 5))] += 1

    by_source_candidates = Counter(it["source"] for it in items)
    by_source_picks = Counter(items[i]["source"] for i in picked)

    zero_pick_sources = {
        source: [it["title"][:90] for it in items
                 if it["source"] == source][:3]
        for source in by_source_candidates
        if not by_source_picks.get(source)
    }

    scores = sorted(
        p["score"] for p in picks
        if isinstance(p.get("score"), (int, float))
        and not isinstance(p.get("score"), bool)
    )

    _run()["selection"][section] = {
        "candidates": n,
        "picks": len(picked),
        "position_quintiles": quintiles,
        "by_source_candidates": dict(by_source_candidates),
        "by_source_picks": dict(by_source_picks),
        "zero_pick_sources": zero_pick_sources,
        "score_min": scores[0] if scores else None,
        "score_median": statistics.median(scores) if scores else None,
        "score_max": scores[-1] if scores else None,
    }


@_safe
def section_result(section, quota, selected, distinct_events,
                   taken_elsewhere, ran_out):
    _run()["sections"][section] = {
        "quota": quota,
        "published": len(selected),
        "distinct_events": distinct_events,
        "taken_by_earlier_section": taken_elsewhere,
        "stopped": "ran out of events" if ran_out else "quota reached",
    }


# ------------------------------------------------------------------ clusters

@_safe
def clusters(report):
    _run()["clusters"] = report


# ----------------------------------------------------------------- published

def _previous_published():
    rows = []
    if not os.path.isdir(RUNS_DIR):
        return rows

    cutoff = _now() - dt.timedelta(days=CROSS_DAY_DAYS)
    seen_titles = set()

    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(RUNS_DIR, name), encoding="utf-8") as f:
                data = json.load(f)
            started = dt.datetime.fromisoformat(data["started"])
        except Exception:
            continue

        if started < cutoff:
            break

        for p in data.get("published", []):
            title = p.get("title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            rows.append({
                "date": data["started"][:10],
                "title": title,
                "summary": p.get("summary", ""),
            })

    return rows


def _cross_day(rows):
    """Log-only: which of today's stories resemble recently published ones."""
    previous = _previous_published()
    if not rows or not previous:
        return []

    # Lazy import: dedupe_v2 imports this module at load time.
    from dedupe_v2 import cosine, make_vectors

    docs = [
        {"title": r["title"], "summary": r.get("summary", ""), "blurb": ""}
        for r in rows + previous
    ]
    vectors = make_vectors(docs)
    found = []

    for i, today in enumerate(rows):
        for j, old in enumerate(previous):
            sim = cosine(vectors[i], vectors[len(rows) + j])
            if sim >= CROSS_DAY_THRESHOLD:
                found.append({
                    "similarity": round(sim, 2),
                    "today": today["title"][:90],
                    "earlier": old["title"][:90],
                    "earlier_date": old["date"],
                })

    found.sort(key=lambda r: -r["similarity"])
    return found[:20]


@_safe
def published(chosen):
    now = _now()
    rows = []

    for item in chosen:
        age = None
        if item.get("ts"):
            try:
                ts = dt.datetime.fromisoformat(item["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                age = round((now - ts).total_seconds() / 3600, 1)
            except ValueError:
                pass

        rows.append({
            "section": item.get("section"),
            "source": item.get("source"),
            "title": item.get("title"),
            "link": item.get("link"),
            "score": item.get("score"),
            "star": bool(item.get("open")),
            "age_hours": age,
            "summary": item.get("summary", ""),
        })

    _run()["published"] = rows
    _run()["repeats"] = _cross_day(rows)


# -------------------------------------------------------------------- report

def _pct(x):
    return f"{x:>4.0%}"


def _section_order(run):
    order = list(run["sections"])
    order += [s for s in run["selection"] if s not in order]
    return order


def render_report(run):
    L = []
    add = L.append
    u = run["usage"]
    c = run["counters"]

    add(f"DIGEST RUN {run['started']}  status={run['status']}  "
        f"model={run['model']}  {run.get('seconds', '?')}s")
    if run.get("error"):
        add(f"ERROR: {run['error']}")
    for key, value in run["notes"].items():
        add(f"note: {key} = {value}")

    cost = f", cost ${u['cost']:.4f}" if u.get("cost") else ""
    add(f"\nUSAGE: {u['calls']} calls, {u['prompt_tokens']:,} tokens in, "
        f"{u['completion_tokens']:,} out{cost}")
    for label, v in sorted(u["by_label"].items()):
        add(f"   {label:<26}{v['calls']:>3} calls "
            f"{v['prompt_tokens']:>9,} in {v['completion_tokens']:>7,} out")
    add(f"RETRIES: http={c.get('http_retries', 0)} "
        f"connection={c.get('connection_retries', 0)} "
        f"unparseable_reply={c.get('parse_retries', 0)}")

    # ---- sections
    add("\nSECTIONS (events = distinct events among this section's picks "
        "after clustering)")
    add(f"  {'section':<13}{'cand':>5}{'picks':>6}{'events':>7}"
        f"{'pub/quota':>11}  stopped")
    for s in _section_order(run):
        sel = run["selection"].get(s, {})
        res = run["sections"].get(s, {})
        pubq = f"{res.get('published', '?')}/{res.get('quota', '?')}"
        stopped = res.get("stopped", "")
        taken = res.get("taken_by_earlier_section")
        if taken:
            stopped += f" ({taken} events already taken by an earlier section)"
        add(f"  {s:<13}{sel.get('candidates', '-'):>5}"
            f"{sel.get('picks', '-'):>6}"
            f"{str(res.get('distinct_events', '-')):>7}{pubq:>11}  {stopped}")

    # ---- position
    add("\nPICK POSITION (share of picks from each fifth of the list sent, "
        "front -> back)")
    for s in _section_order(run):
        sel = run["selection"].get(s)
        if not sel or not sel["picks"]:
            continue
        q = sel["position_quintiles"]
        total = sum(q) or 1
        line = f"  {s:<13}" + " ".join(_pct(x / total) for x in q)
        if sel["picks"] >= 10 and (q[0] + q[1]) / total >= POSITION_BIAS_SHARE:
            line += "   <- front-loaded: possible position bias"
        add(line)

    # ---- source yield
    sent, picks, pubs = Counter(), Counter(), Counter()
    for f in run["feeds"]:
        sent[(f["section"], f["source"])] += f["sent"]
    for s, sel in run["selection"].items():
        for source, n in sel["by_source_picks"].items():
            picks[(s, source)] += n
    for p in run["published"]:
        pubs[(p["section"], p["source"])] += 1

    add("\nSOURCE YIELD (sent -> picked -> published)")
    for key in sorted(set(sent) | set(picks) | set(pubs)):
        flag = ("   <- zero picks"
                if sent[key] >= 10 and picks[key] == 0 else "")
        add(f"  {key[0]:<12} {key[1]:<14}{sent[key]:>4} -> "
            f"{picks[key]:>3} -> {pubs[key]:>3}{flag}")

    # ---- zero-pick samples
    add("\nZERO-PICK SOURCES (first titles, to judge: junk feed or ignored?)")
    any_zero = False
    for s in _section_order(run):
        sel = run["selection"].get(s, {})
        for source, titles in sel.get("zero_pick_sources", {}).items():
            any_zero = True
            add(f"  {s} | {source}")
            for t in titles:
                add(f"      {t}")
    if not any_zero:
        add("  (none)")

    # ---- clusters
    cl = run.get("clusters")
    add("\nCLUSTERING")
    if not cl:
        add("  (no clustering recorded)")
    else:
        add(f"  AI clustering: {cl.get('ai_clustering')}   "
            f"{cl.get('candidates')} candidates -> {cl.get('groups')} groups   "
            f"sizes {cl.get('size_histogram')}")
        add(f"  response repairs: {run.get('repairs') or 'none'}")
        add(f"  auto-merged by similarity: "
            f"{len(cl.get('auto_merged', []))}")
        for m in cl.get("auto_merged", []):
            add(f"      {m['similarity']:.2f}  {m['a']}  |  {m['b']}")

        add(f"  WEAK LINKS ({len(cl.get('weak_links', []))}): cluster members "
            "barely similar to the rest; review for false merges")
        for w in cl.get("weak_links", []):
            add(f"      {w['similarity']:.2f} {w['cluster']} "
                f"(size {w['cluster_size']}) {w['source']}: {w['title']}")

        add(f"  CROSS-SECTION clusters ({len(cl.get('cross_section', []))}): "
            "one event seen in several sections; one copy is kept")
        for x in cl.get("cross_section", []):
            add(f"      {x['cluster']} {x['sections']}")

        add("  MULTI-MEMBER clusters:")
        for g in cl.get("multi_member_clusters", []):
            add(f"      {g['cluster']} ({len(g['members'])})")
            for m in g["members"]:
                add(f"          {m['source']}: {m['title']}")

        add(f"  LEXICAL AUDIT (similar but not merged): "
            f"{len(cl.get('audit_pairs', []))} shown")
        for a in cl.get("audit_pairs", []):
            add(f"      {a['similarity']:.2f}  {a['a']}  |  {a['b']}")

    # ---- scores
    add("\nSCORES (min / median / max)")
    for s in _section_order(run):
        sel = run["selection"].get(s, {})
        pub = [p["score"] for p in run["published"]
               if p["section"] == s and isinstance(p.get("score"), (int, float))]
        picks_part = (f"{sel.get('score_min')} / {sel.get('score_median')} / "
                      f"{sel.get('score_max')}") if sel else "-"
        pub_part = (f"{min(pub)} / {statistics.median(pub)} / {max(pub)}"
                    if pub else "-")
        add(f"  {s:<13} picks {picks_part:<16} published {pub_part}")

    # ---- published
    add(f"\nPUBLISHED ({len(run['published'])}) section | score | age | source | title")
    for p in run["published"]:
        age = f"{p['age_hours']}h" if p.get("age_hours") is not None else "?"
        star = "*" if p.get("star") else " "
        add(f" {star} {p['section']:<12} {str(p['score']):>3} {age:>7}  "
            f"{p['source']}: {p['title']}")

    # ---- repeats
    add(f"\nPOSSIBLE REPEATS OF THE LAST {CROSS_DAY_DAYS} DAYS "
        f"(similarity >= {CROSS_DAY_THRESHOLD}; may be updates, not repeats)")
    if not run["repeats"]:
        add("  (none found, or no earlier runs on file)")
    for r in run["repeats"]:
        add(f"  {r['similarity']:.2f}  {r['today']}  |  "
            f"{r['earlier_date']}: {r['earlier']}")

    # ---- feed problems
    problems = [f for f in run["feeds"] if f["error"] or f["sent"] == 0]
    add(f"\nFEED PROBLEMS ({len(problems)})")
    for f in problems:
        why = f["error"] or "0 items sent (empty, all seen, or out of window)"
        add(f"  {f['section']} | {f['source']}: {why}")

    # ---- events
    if run["events"]:
        add("\nEVENTS")
        for e in run["events"]:
            add(f"  {e['kind']}: {e['detail']}")

    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- finish

def _summary_line(run):
    sections = {}
    for s in _section_order(run):
        sel = run["selection"].get(s, {})
        res = run["sections"].get(s, {})
        sections[s] = {
            "candidates": sel.get("candidates"),
            "picks": sel.get("picks"),
            "events": res.get("distinct_events"),
            "published": res.get("published"),
            "quota": res.get("quota"),
        }

    cl = run.get("clusters") or {}
    return {
        "started": run["started"],
        "status": run["status"],
        "model": run["model"],
        "seconds": run.get("seconds"),
        "usage": {k: run["usage"][k]
                  for k in ("calls", "prompt_tokens",
                            "completion_tokens", "cost")},
        "retries": run["counters"],
        "ai_clustering": cl.get("ai_clustering"),
        "groups": cl.get("groups"),
        "repairs": run.get("repairs"),
        "weak_links": len(cl.get("weak_links", [])),
        "feed_problems": sum(1 for f in run["feeds"]
                             if f["error"] or f["sent"] == 0),
        "published_total": len(run["published"]),
        "possible_repeats": len(run["repeats"]),
        "notes": run["notes"],
        "sections": sections,
    }


def _write(path, text, mode="w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if mode == "a":
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


@_safe
def finish_run(status="ok", error=None):
    run = _run()
    run["status"] = status
    run["error"] = error
    run["finished"] = _now().isoformat()
    run["seconds"] = round(time.time() - run.get("_t0", time.time()), 1)

    clean = {k: v for k, v in run.items() if not k.startswith("_")}
    stamp = dt.datetime.fromisoformat(run["started"]).strftime("%Y-%m-%dT%H%M%S%fZ")

    _write(os.path.join(RUNS_DIR, f"run-{stamp}.json"),
           json.dumps(clean, ensure_ascii=False, indent=1, default=str))
    _write(JSONL_PATH,
           json.dumps(_summary_line(clean), ensure_ascii=False,
                      default=str) + "\n", mode="a")
    _write(REPORT_PATH, render_report(clean))

    # keep the newest KEEP_RUN_FILES detail files
    files = sorted(n for n in os.listdir(RUNS_DIR) if n.endswith(".json"))
    for old in files[:-KEEP_RUN_FILES]:
        os.remove(os.path.join(RUNS_DIR, old))

    print(f"\nDiagnostics written to {DIAG_DIR}/ "
          f"(latest.txt, runs.jsonl, runs/run-{stamp}.json)")

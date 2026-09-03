"""Generate a minimal browser UI for a packaged FastAPI app, so a human can exercise the
generated controllers and models by hand.

==========================  WHY THIS IS BUILT FROM CONTRACTS  ==========================
The page is derived ENTIRELY from ``ir['planning']['contracts']`` — never from the generated
code. Everything it needs is already deterministic:

  * the route comes from ``contracts.route_for(file_path)``, the same function the packager
    and the certification oracle use, so the form cannot drift from where the router is
    actually mounted;
  * the parameter names come from the signature;
  * the prefilled request body is the contract's ``example_inputs_json``;
  * the documented reply is its ``expected_return_json``.

So this adds no LLM call, no cost, and no clean-room concern: it reads the same spec-derived
contract every generator reads. It is a debugging aid, never an input to any agent, and
nothing it produces is scored.
=======================================================================================

The result is one self-contained ``app/static/index.html`` — no build step, no CDN (the
generated app may run with no network), and it talks to its own origin, so no CORS setup.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from src.cleanroom.utils.contracts import route_for


def _params(signature: str) -> list[str]:
    """Parameter names from a ``def f(a: int, b: str) -> X`` signature."""
    m = re.search(r"\((.*)\)", signature or "", re.S)
    if not m:
        return []
    names: list[str] = []
    depth = 0
    current = ""
    for ch in m.group(1):                      # split on commas at depth 0 (skip generics)
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            names.append(current)
            current = ""
        else:
            current += ch
    names.append(current)
    out = []
    for raw in names:
        name = raw.split(":")[0].split("=")[0].strip()
        if name and name not in ("self", "cls", "*", "/"):
            out.append(name)
    return out


def _pretty(raw: str, fallback: str) -> str:
    """Pretty-print a JSON string for the textarea, leaving unparseable text alone."""
    try:
        return json.dumps(json.loads(raw), indent=2)
    except (ValueError, TypeError):
        return raw or fallback


def _endpoints(ir: dict) -> list[dict]:
    """One entry per contract, in the planner's dependency order."""
    req_text: dict[str, str] = {}
    for feature in ir.get("features", []) or []:
        for req in feature.get("functional_requirements", []) or []:
            req_text[req["id"]] = (req.get("description") or "").strip()

    out = []
    for c in (ir.get("planning") or {}).get("contracts", []) or []:
        params = _params(c.get("signature", ""))
        body = _pretty(c.get("example_inputs_json", ""),
                       json.dumps({p: None for p in params}, indent=2))
        out.append({
            "fr_id": c.get("fr_id", ""),
            "feature_id": c.get("feature_id", ""),
            "layer": c.get("mvc_layer", ""),
            "route": route_for(c.get("file_path", "")),
            "signature": c.get("signature", ""),
            "docstring": (c.get("docstring") or "").strip(),
            "requirement": req_text.get(c.get("fr_id", ""), ""),
            "params": params,
            "body": body,
            "expected": _pretty(c.get("expected_return_json", ""), ""),
            "error_mode": c.get("error_mode", "raise"),
            "failure": _pretty(c.get("failure_inputs_json", ""), ""),
        })
    return out


def build_ui(ir: dict, app_dir: Path, pages: dict[str, str] | None = None) -> Path | None:
    """Write the app's static UI. Returns the front page, or None when there is nothing to show.

    Without ``pages`` this is unchanged: the deterministic contract console becomes
    ``index.html``, as before.

    With ``pages`` — ``{feature_id: html}`` from the Frontend Agent — each generated page is
    written as ``feature_<id>.html``, the console moves to ``console.html`` (it stays useful for
    debugging, and it is the fallback when a feature's page failed to generate), and
    ``index.html`` becomes a deterministic shell linking them. The shell is built here rather
    than by an agent so the navigation cannot point at a page that does not exist.
    """
    # Falls back to the IR so a caller cannot silently lose the generated pages by omitting
    # the argument — the pages live on the same ir that is already being passed in.
    if pages is None:
        pages = (ir.get("generated_frontend") or {}).get("pages") or None

    endpoints = _endpoints(ir)
    if not endpoints and not pages:
        return None

    static = Path(app_dir) / "static"
    static.mkdir(parents=True, exist_ok=True)

    if not pages:
        page = static / "index.html"
        page.write_text(_render(endpoints))
        return page

    (static / "console.html").write_text(_render(endpoints))
    written: list[tuple[str, str, str]] = []          # (feature_id, name, href)
    names = {str(f.get("id")): f.get("name", "") for f in ir.get("features", []) or []}
    for feature_id, page_html in sorted(pages.items(), key=lambda kv: _numeric(kv[0])):
        href = f"feature_{str(feature_id).replace('.', '_')}.html"
        (static / href).write_text(page_html)
        written.append((feature_id, names.get(str(feature_id), ""), href))

    index = static / "index.html"
    index.write_text(_render_shell(written))
    return index


def _numeric(feature_id: str):
    """Sort feature ids the way a reader expects (2.10 after 2.9, not before)."""
    return [int(p) if p.isdigit() else p for p in str(feature_id).split(".")]


_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light dark; }
body { margin:0; font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
        background:#f6f7f9; color:#14161a; }
@media (prefers-color-scheme: dark) { body { background:#111316; color:#e8eaed; } }
header { padding:28px 24px 12px; }
h1 { margin:0 0 4px; font-size:20px; letter-spacing:-.01em; }
p.sub { margin:0; opacity:.65; font-size:13px; }
ul { list-style:none; margin:18px 24px 40px; padding:0; display:grid; gap:10px;
      grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); max-width:1100px; }
a.card { display:block; padding:14px 16px; border-radius:10px; text-decoration:none;
          background:#fff; color:inherit; border:1px solid #e3e6ea; }
@media (prefers-color-scheme: dark) { a.card { background:#1a1d21; border-color:#2b2f36; } }
a.card:hover { border-color:#9aa4b2; }
a.card b { display:block; font-size:14px; margin-bottom:2px; }
a.card span { font-size:12px; opacity:.6; }
footer { margin:0 24px 40px; font-size:12px; opacity:.6; }
</style></head><body>
<header><h1>__TITLE__</h1><p class="sub">__COUNT__ feature(s)</p></header>
<ul>__CARDS__</ul>
<footer><a href="console.html">Contract console</a> — every endpoint, prefilled from the spec.</footer>
</body></html>
"""


def _render_shell(written: list[tuple[str, str, str]]) -> str:
    cards = "\n".join(
        f'<li><a class="card" href="{html.escape(href)}">'
        f'<b>{html.escape(name or "Feature " + fid)}</b>'
        f'<span>Feature {html.escape(fid)}</span></a></li>'
        for fid, name, href in written)
    return (_SHELL.replace("__CARDS__", cards)
                  .replace("__COUNT__", str(len(written)))
                  .replace("__TITLE__", "Application"))


def _render(endpoints: list[dict]) -> str:
    cards = "\n".join(_card(i, e) for i, e in enumerate(endpoints))
    layers = sorted({e["layer"] for e in endpoints if e["layer"]})
    filters = "\n".join(
        f'<button class="chip" data-layer="{html.escape(l)}">{html.escape(l)}</button>'
        for l in layers)
    return _PAGE.replace("__CARDS__", cards).replace("__FILTERS__", filters).replace(
        "__COUNT__", str(len(endpoints)))


def _card(i: int, e: dict) -> str:
    esc = html.escape
    failure = ""
    if e["failure"]:
        failure = (f'<button class="mini" data-fill="{i}" '
                   f'data-json="{esc(e["failure"])}">load failure case</button>')
    expected = (f'<div class="expected"><span>documented reply</span>'
                f'<pre>{esc(e["expected"])}</pre></div>') if e["expected"] else ""
    return f"""
<section class="card" data-layer="{esc(e['layer'])}">
  <header>
    <div>
      <span class="fr">FR {esc(e['fr_id'])}</span>
      <span class="badge {esc(e['layer'])}">{esc(e['layer'])}</span>
    </div>
    <code class="route">POST {esc(e['route'])}</code>
  </header>
  <p class="req">{esc(e['requirement'])}</p>
  <details><summary>contract</summary>
    <pre class="sig">{esc(e['signature'])}\n{esc(e['docstring'])}</pre>
  </details>
  <label>request body</label>
  <textarea id="body{i}" spellcheck="false">{esc(e['body'])}</textarea>
  <div class="actions">
    <button class="send" data-i="{i}" data-route="{esc(e['route'])}">Send</button>
    <button class="mini" data-fill="{i}" data-json="{esc(e['body'])}">reset</button>
    {failure}
    <span class="mode">errors: {esc(e['error_mode'])}</span>
  </div>
  {expected}
  <div class="resp" id="resp{i}" hidden></div>
</section>"""


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generated API — manual test console</title>
<style>
  :root{--bg:#0f1115;--card:#171a21;--line:#272b34;--fg:#e6e8ee;--dim:#9aa3b2;
        --accent:#6ea8fe;--ok:#4ade80;--err:#f87171;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
  @media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ec;--fg:#1a1d23;
        --dim:#5f6773;--accent:#2563eb}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif}
  header.top{padding:20px 24px;border-bottom:1px solid var(--line);display:flex;
       align-items:baseline;gap:12px;flex-wrap:wrap}
  h1{font-size:17px;margin:0;font-weight:600}
  .sub{color:var(--dim);font-size:13px}
  .filters{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}
  .chip{background:transparent;border:1px solid var(--line);color:var(--dim);border-radius:99px;
        padding:3px 11px;font-size:12px;cursor:pointer}
  .chip.on{color:var(--fg);border-color:var(--accent)}
  main{padding:20px;display:grid;gap:14px;max-width:900px;margin:0 auto}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  .card header{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
  .fr{font-weight:600;font-size:13px}
  .badge{font-size:11px;padding:1px 8px;border-radius:99px;border:1px solid var(--line);
         color:var(--dim);margin-left:6px}
  .badge.controller{color:var(--accent);border-color:var(--accent)}
  .route{font-family:var(--mono);font-size:12px;color:var(--dim)}
  .req{color:var(--dim);font-size:13px;margin:8px 0}
  details{margin:6px 0}summary{cursor:pointer;color:var(--dim);font-size:12px}
  pre{font-family:var(--mono);font-size:12px;background:var(--bg);border:1px solid var(--line);
      border-radius:6px;padding:8px;overflow-x:auto;margin:6px 0;white-space:pre-wrap}
  label{display:block;font-size:12px;color:var(--dim);margin-top:8px}
  textarea{width:100%;min-height:88px;font-family:var(--mono);font-size:12.5px;background:var(--bg);
           color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;resize:vertical}
  .actions{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
  button.send{background:var(--accent);color:#08131f;border:0;border-radius:6px;padding:6px 16px;
        font-weight:600;cursor:pointer}
  .mini{background:transparent;border:1px solid var(--line);color:var(--dim);border-radius:6px;
        padding:5px 10px;font-size:12px;cursor:pointer}
  .mode{font-size:11px;color:var(--dim);margin-left:auto}
  .expected span{font-size:11px;color:var(--dim)}
  .resp{margin-top:10px;border-top:1px solid var(--line);padding-top:8px}
  .status{font-family:var(--mono);font-size:12px}
  .status.ok{color:var(--ok)}.status.err{color:var(--err)}
</style></head><body>
<header class="top">
  <h1>Generated API — manual test console</h1>
  <span class="sub">__COUNT__ endpoint(s), built from the planner's contracts</span>
  <div class="filters"><button class="chip on" data-layer="">all</button>__FILTERS__</div>
</header>
<main>__CARDS__</main>
<script>
document.querySelectorAll('.chip').forEach(function(chip){
  chip.addEventListener('click', function(){
    document.querySelectorAll('.chip').forEach(function(c){c.classList.remove('on');});
    chip.classList.add('on');
    var want = chip.dataset.layer;
    document.querySelectorAll('.card').forEach(function(card){
      card.hidden = want !== '' && card.dataset.layer !== want;
    });
  });
});

document.querySelectorAll('[data-fill]').forEach(function(btn){
  btn.addEventListener('click', function(){
    document.getElementById('body' + btn.dataset.fill).value = btn.dataset.json;
  });
});

document.querySelectorAll('.send').forEach(function(btn){
  btn.addEventListener('click', async function(){
    var i = btn.dataset.i, box = document.getElementById('resp' + i);
    var raw = document.getElementById('body' + i).value;
    box.hidden = false;
    var payload;
    try { payload = raw.trim() ? JSON.parse(raw) : {}; }
    catch (e) {
      box.innerHTML = '<div class="status err">request body is not valid JSON: ' + e.message + '</div>';
      return;
    }
    btn.disabled = true; box.innerHTML = '<div class="status">sending…</div>';
    var started = performance.now();
    try {
      var res = await fetch(btn.dataset.route, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      var text = await res.text(), shown = text;
      try { shown = JSON.stringify(JSON.parse(text), null, 2); } catch (e) {}
      var ms = Math.round(performance.now() - started);
      box.innerHTML = '<div class="status ' + (res.ok ? 'ok' : 'err') + '">HTTP ' +
        res.status + ' · ' + ms + ' ms</div><pre></pre>';
      box.querySelector('pre').textContent = shown;
    } catch (e) {
      box.innerHTML = '<div class="status err">request failed: ' + e.message +
        '</div><pre>Is the app running? Start it with:  python -m app</pre>';
    } finally { btn.disabled = false; }
  });
});
</script></body></html>
"""

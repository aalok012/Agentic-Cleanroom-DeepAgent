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


def build_ui(ir: dict, app_dir: Path) -> Path | None:
    """Write ``app/static/index.html``. Returns the path, or None when there is nothing to show."""
    endpoints = _endpoints(ir)
    if not endpoints:
        return None

    static = Path(app_dir) / "static"
    static.mkdir(parents=True, exist_ok=True)
    page = static / "index.html"
    page.write_text(_render(endpoints))
    return page


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

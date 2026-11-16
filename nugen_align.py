#!/usr/bin/env python3
"""
Nugen take-home: end-to-end domain alignment via the Nugen API.

  upload docs -> generate benchmark -> create alignment -> wait -> deploy -> chat app

Setup (Windows PowerShell):
    pip install requests pypdf
    $env:NUGEN_API_KEY = "..."        # from https://platform.nugen.in (never hard-code it)

Usage:
    python nugen_align.py spec                                  # show API schema (sanity check)
    python nugen_align.py run  docs                             # full pipeline on a folder
    python nugen_align.py chat                                  # chat with the deployed model
    python nugen_align.py ask "your question"                   # one-shot question

Progress is saved to .nugen_state.json, so if anything stops partway,
re-running `run` resumes from the last finished step instead of paying twice.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.nugen.in"
STATE_FILE = Path(".nugen_state.json")
BASE_MODEL = os.environ.get("NUGEN_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")  # small = cheap
NUM_QUESTIONS = int(os.environ.get("NUGEN_NUM_QUESTIONS", "20"))

DONE = {"READY"}                 # ResourceStatus: CREATED/QUEUED/PROCESSING/READY/FAILED/STOPPED
FAILED = {"FAILED", "STOPPED"}


def headers():
    key = os.environ.get("NUGEN_API_KEY")
    if not key:
        sys.exit('Set your key first:  $env:NUGEN_API_KEY = "..."')
    return {"Authorization": f"Bearer {key}"}


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def call(method, path, **kw):
    r = requests.request(method, f"{BASE_URL}{path}", headers=headers(), timeout=120, **kw)
    if not r.ok:
        sys.exit(f"{method} {path} -> {r.status_code}\n{r.text}")
    return r.json() if r.content else {}


def wait(path, label, interval=15, timeout=6 * 3600):
    """Poll a status endpoint until it reaches a terminal state."""
    start = time.time()
    while True:
        data = call("GET", path)
        status = str(data.get("status") or "").upper()
        print(f"  [{label}] {status or data}  ({int(time.time() - start)}s)")
        if status in DONE:
            return data
        if status in FAILED:
            sys.exit(f"{label} failed: {json.dumps(data, indent=2)}")
        if time.time() - start > timeout:
            sys.exit(f"{label} timed out; re-run later to resume.")
        time.sleep(interval)


# ---------------------------------------------------------------- PDF -> text
def to_text_files(args):
    """Collect files (folders allowed); convert PDFs to .txt in a 'docs_txt' folder."""
    paths = []
    for a in args:
        p = Path(a)
        paths += sorted(x for x in p.iterdir() if x.is_file()) if p.is_dir() else [p]
    out_dir = Path("docs_txt")
    out_dir.mkdir(exist_ok=True)
    result = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            from pypdf import PdfReader  # pip install pypdf
            text = "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
            txt = out_dir / (p.stem + ".txt")
            txt.write_text(text, encoding="utf-8")
            p = txt
        elif p.suffix.lower() not in {".txt", ".md", ".jsonl"}:
            print(f"   skipping {p.name} (not PDF/text)")
            continue
        words = len(p.read_text(encoding="utf-8", errors="ignore").split())
        print(f"   {p.name}: {words} words")
        if words < 500 and p.suffix.lower() != ".jsonl":
            sys.exit(f"{p.name} has under 500 words; Nugen will refuse it. Remove or replace it.")
        result.append(p)
    return result


# ---------------------------------------------------------------- pipeline
def run(files):
    st = load_state()

    # 1. Upload documents (Nugen wants plain text, >= 500 words per file)
    if "document_ids" not in st:
        paths = to_text_files(files)
        if not paths:
            sys.exit("Pass at least one document (or a folder of documents) to upload.")
        print(f"1. Uploading {len(paths)} document(s)...")
        ids = []
        for p in paths:
            with p.open("rb") as fh:
                res = call("POST", "/api/v3/documents/create",
                           data={"names": [p.stem]},
                           files=[("files", (p.name, fh, "text/plain"))])
            print(f"   {p.name}: {res}")
            ids += res["document_ids"]
        st["document_ids"] = ids
        save_state(st)
    for did in st["document_ids"]:
        wait(f"/api/v3/documents/{did}/status", f"doc {did[-6:]}")

    # 2. Generate benchmark (used to auto-evaluate the aligned model)
    if "benchmark_id" not in st:
        print(f"2. Generating a {NUM_QUESTIONS}-question benchmark...")
        res = call("POST", "/api/v3/benchmarks/create",
                   json={"document_ids": st["document_ids"], "n_samples": NUM_QUESTIONS,
                         "benchmark_name": "Health insurance QA"})
        st["benchmark_id"] = res["benchmark_id"]
        save_state(st)
    wait(f"/api/v3/benchmarks/{st['benchmark_id']}/status", "benchmark")
    bench = call("GET", f"/api/v3/benchmarks/{st['benchmark_id']}/data")
    Path("benchmark.json").write_text(json.dumps(bench.get("samples", bench), indent=2, ensure_ascii=False))
    print("   Saved benchmark.json (review it: these Q&As define what 'good' means).")

    # 3. Create alignment project
    if "alignment_id" not in st:
        print(f"3. Creating alignment on {BASE_MODEL}...")
        res = call("POST", "/api/v3/alignment-projects/create", json={
            "alignment_name": "joy-domain-alignment",
            "base_model_id": BASE_MODEL,
            "document_ids": st["document_ids"],
            "benchmark_id": st["benchmark_id"],
            "description": "Nugen take-home exercise",
        })
        st["alignment_id"] = res["alignment_id"]
        save_state(st)
    wait(f"/api/v3/alignment-projects/{st['alignment_id']}/status", "alignment", interval=30)

    # 4. Find the aligned model and deploy it
    if "model_id" not in st:
        models = call("GET", "/api/v3/models/aligned").get("domain_aligned_models", [])
        mine = [m for m in models if st["alignment_id"] in json.dumps(m)] or models
        if not mine:
            sys.exit("No aligned model found yet; re-run in a few minutes.")
        st["model_id"] = mine[0]["model_id"]
        save_state(st)
    print(f"4. Deploying {st['model_id']}...")
    if "deployed_model_id" not in st:
        res = call("POST", f"/api/v3/models/{st['model_id']}/deployment")
        st["deployed_model_id"] = res.get("model_id") or st["model_id"]  # may carry a -deployed suffix
        save_state(st)
    start = time.time()
    while True:  # statuses: UNDEPLOYED / DEPLOYING / DEPLOYED / UNDEPLOYING
        d = call("GET", f"/api/v3/models/{st['model_id']}/deployment/status")
        print(f"  [deployment] {d.get('status')}  ({int(time.time() - start)}s)")
        if d.get("status") == "DEPLOYED":
            break
        if d.get("status") == "UNDEPLOYED" and d.get("error"):
            st.pop("deployed_model_id", None)
            save_state(st)
            sys.exit(f"Deploy failed: {d['error']}  (re-run to retry)")
        time.sleep(15)

    # 5. Evaluation results (if the benchmark eval ran)
    model = call("GET", f"/api/v3/models/{st['model_id']}")
    eval_id = (model.get("evaluation_data") or {}).get("evaluation_id")
    if eval_id:
        wait(f"/api/v3/evaluations/{eval_id}/status", "evaluation")
        results = call("GET", f"/api/v3/evaluations/{eval_id}/results")
        Path("evaluation.json").write_text(json.dumps(results, indent=2))
        print("5. Saved evaluation.json")

    print(f"\nDone. Model {st['deployed_model_id']} is live. Try:  python {sys.argv[0]} chat")


# ---------------------------------------------------------------- the "build something" part
SYSTEM = ("You are a domain assistant. Answer only from your domain knowledge. "
          "If you are not sure, say so instead of guessing.")


def stream_answer(messages):
    st = load_state()
    model = st.get("deployed_model_id")
    if not model:
        sys.exit("No deployed model yet. Run the pipeline first.")
    r = requests.post(f"{BASE_URL}/api/v3/inference/chat/completions", headers=headers(), stream=True,
                      timeout=120, json={"model": model, "messages": messages,
                                         "max_tokens": 400, "temperature": 0.2, "stream": True})
    if not r.ok:
        sys.exit(f"Inference error {r.status_code}: {r.text}")
    out = []
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        chunk = line[5:].strip()
        if chunk == b"[DONE]":
            break
        try:
            delta = json.loads(chunk)["choices"][0].get("delta", {}).get("content") or ""
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        print(delta, end="", flush=True)
        out.append(delta)
    print()
    return "".join(out)


def chat():
    messages = [{"role": "system", "content": SYSTEM}]
    print("Chatting with your aligned model (Ctrl+C or 'exit' to quit).")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        messages.append({"role": "user", "content": q})
        print("model> ", end="")
        messages.append({"role": "assistant", "content": stream_answer(messages)})


def spec():
    """Print the API schema for the endpoints this script relies on."""
    doc = requests.get(f"{BASE_URL}/openapi-public.json", timeout=60).json()
    comps = doc.get("components", {}).get("schemas", {})

    def resolve(s, depth=0):
        if isinstance(s, dict):
            if "$ref" in s and depth < 4:
                return resolve(comps.get(s["$ref"].split("/")[-1], {}), depth + 1)
            return {k: resolve(v, depth) for k, v in s.items()}
        if isinstance(s, list):
            return [resolve(x, depth) for x in s]
        return s

    wanted = ["documents/create", "documents/{", "benchmarks/create",
              "alignment-projects/create", "deployment", "models/base"]
    for path, ops in doc.get("paths", {}).items():
        if any(w in path for w in wanted):
            for method, op in ops.items():
                print(f"\n=== {method.upper()} {path}")
                body = op.get("requestBody", {}).get("content", {})
                for ctype, c in body.items():
                    print(f"request ({ctype}):", json.dumps(resolve(c.get("schema", {})), indent=1)[:2500])
                ok = op.get("responses", {}).get("200", {}).get("content", {})
                for ctype, c in ok.items():
                    print("response:", json.dumps(resolve(c.get("schema", {})), indent=1)[:1500])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "spec":
        spec()
    elif cmd == "run":
        run(sys.argv[2:])
    elif cmd == "chat":
        chat()
    elif cmd == "ask" and len(sys.argv) > 2:
        stream_answer([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": " ".join(sys.argv[2:])}])
    else:
        print(__doc__)
"""
Insurance Q&A with sources: retrieval (BM25 over your documents) + your aligned Nugen model.

    python rag.py                          # interactive
    python rag.py "your question"          # one question
    python rag.py --compare "question"     # aligned model alone vs. aligned model + documents
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

import nugen_align as n

DOCS_DIR = Path("docs_txt")
CHUNK_WORDS, OVERLAP, TOP_K = 150, 50, 4
STOP = set("a an the of to in on for and or is are be by with as at from that this it its any shall "
           "will may can under such which not no if per than other".split())

SYSTEM = ("You are an assistant for Indian health insurance. Answer ONLY from the numbered context "
          "passages. Cite passages like [1]. Mention clause numbers or periods exactly as written. "
          "If the context does not contain the answer, say you could not find it in the documents.")


# Insurance wording differs from everyday wording ("rejected" vs "contestable"/"repudiated")
SYNONYMS = {
    "reject": ["contest", "contestable", "repudiat", "deni", "deny"],
    "deny": ["reject", "repudiat", "contest"],
    "cancel": ["terminat", "cancellation"],
    "wait": ["waiting"],
    "ped": ["pre", "existing", "disease"],
    "refund": ["return", "premium"],
    "complaint": ["grievance", "ombudsman"],
    "approve": ["authoriz", "authoris", "decide", "pre"],
    "long": ["hour", "days", "within"],
    "time": ["hour", "days", "within"],
}


def stem(t):
    for suf in ("ation", "ing", "ies", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def tokens(text):
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def query_tokens(text):
    q = tokens(text)
    return list(dict.fromkeys(q + [stem(s) for t in q for s in SYNONYMS.get(t, [])]))


def load_chunks():
    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        sys.exit("No docs_txt folder found. Run `python nugen_align.py run docs` first.")
    chunks = []
    for f in files:
        words = f.read_text(encoding="utf-8", errors="ignore").split()
        for i in range(0, max(len(words) - OVERLAP, 1), CHUNK_WORDS - OVERLAP):
            text = " ".join(words[i:i + CHUNK_WORDS])
            chunks.append({"source": f.stem, "text": text, "toks": tokens(text)})
    return chunks


class BM25:
    def __init__(self, chunks, k1=0.9, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(c["toks"]) for c in chunks]
        self.avg = sum(len(c["toks"]) for c in chunks) / len(chunks)
        df = Counter(t for c in chunks for t in set(c["toks"]))
        N = len(chunks)
        self.idf = {t: math.log(1 + (N - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, k=TOP_K):
        q = query_tokens(query)
        total = sum(self.idf.get(t, 0) ** 2 for t in q) or 1
        scored = []
        for i, tf in enumerate(self.tf):
            L = len(self.chunks[i]["toks"])
            matched = [t for t in q if t in tf]
            # idf squared: rare, specific words ("moratorium") outweigh common ones ("claim")
            s = sum(self.idf.get(t, 0) ** 2 * tf[t] * (self.k1 + 1) /
                    (tf[t] + self.k1 * (1 - self.b + self.b * L / self.avg)) for t in matched)
            coverage = sum(self.idf.get(t, 0) ** 2 for t in matched) / total
            if s > 0:
                scored.append((s * (1 + 2 * coverage), i))
        hits, seen = [], set()
        for _, i in sorted(scored, reverse=True):
            key = self.chunks[i]["text"][:120]
            if key not in seen:  # overlapping chunks can repeat the same passage
                seen.add(key)
                hits.append(self.chunks[i])
            if len(hits) == k:
                break
        return hits

    def best_excerpt(self, text, query, width=3):
        """The few sentences of a passage that match the question best (keeps a small model focused)."""
        q = set(query_tokens(query))
        sents = [s for s in re.split(r"(?<=[.;:])\s+", text) if s.strip()]
        score = [sum(self.idf.get(t, 0) ** 2 for t in q & set(tokens(s))) for s in sents]
        best = max(range(len(sents)), key=score.__getitem__)
        return " ".join(sents[max(best - 1, 0):best - 1 + width])


def answer(index, question):
    hits, excerpts = [], []
    for h in index.search(question, k=TOP_K * 3):
        e = index.best_excerpt(h["text"], question)
        if e[:100] not in {x[:100] for x in excerpts}:  # skip repeats of the same clause
            hits.append(h)
            excerpts.append(e)
        if len(hits) == TOP_K:
            break
    context = "\n\n".join(f"[{i}] ({h['source']})\n{e}" for i, (h, e) in enumerate(zip(hits, excerpts), 1))
    print("answer> ", end="")
    n.stream_answer([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    print("\nsources:")
    for i, (h, e) in enumerate(zip(hits, excerpts), 1):
        print(f"  [{i}] {h['source']}: {e[:300]}")


def main():
    index = BM25(load_chunks())
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        q = " ".join(args[1:])
        print("=== Aligned model alone ===")
        n.stream_answer([{"role": "user", "content": q}])
        print("\n=== Aligned model + your documents ===")
        answer(index, q)
    elif args:
        answer(index, " ".join(args))
    else:
        print("Ask about your insurance documents ('exit' to quit).")
        while (q := input("\nyou> ").strip()).lower() not in {"exit", "quit"}:
            if q:
                answer(index, q)


if __name__ == "__main__":
    main()
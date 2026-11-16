Health Insurance Q&A on a Nugen Domain-Aligned Model
Hands-on exercise for Nugen Intelligence: align a model to a domain using the Nugen API, deploy it, and build something with it.
Domain and documents
Indian health insurance:
Arogya Sanjeevani Policy wording (Go Digit)
IRDAI Master Circular on Health Insurance Business (29 May 2024)
IRDAI Master Circular on Protection of Policyholders' Interests (5 Sept 2024)
What's here
File	What it does
`nugen_align.py`	End-to-end pipeline via the Nugen API: PDF → text, document upload, benchmark generation, alignment (Qwen2.5-1.5B-Instruct), deployment, inference. Saves progress so a failed step can be resumed.
`rag.py`	Q&A app: finds the most relevant clauses in the documents (BM25 with insurance synonyms), sends them to the aligned model, and prints the answer with its sources.
`benchmark.json`	The 20 questions Nugen generated from the documents.
Run it
```powershell
pip install requests pypdf
$env:NUGEN_API_KEY = "your-key"
python nugen_align.py run docs        # align + deploy
python rag.py --compare "Can a health insurance claim be rejected after the moratorium period?"
```
Result
Question	Aligned model alone	Aligned model + retrieved clauses
Can a claim be rejected after the moratorium period?	"Yes, it can be rejected" (wrong)	"No, it cannot be rejected" [cites Master Circular §13: 60 months, except established fraud]
How long to approve a cashless request?	n/a	"Not more than one hour of receipt of request"
The aligned model alone was inconsistent on exact regulatory facts. Adding retrieval fixed this and made every answer checkable against its source.
Issues found in the platform
First alignment failed ~90s after creation: "no activity for 1 hours". `created_at` was in UTC while `updated_at` was in IST. The inactivity check appears to compare mixed time zones. The Models page also shows the failed run as created at 11:28 am when it was started at about 4:58 pm IST.
Benchmark evaluation stayed in PROCESSING for about an hour, then failed with `error: null`. Its timestamps were also mixed between IST and UTC.

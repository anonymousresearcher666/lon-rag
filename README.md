# LON-RAG Anonymous Reproducibility Package

This is a **code-only, anonymous, lightweight** artifact for reproducing the
experiments described in the accompanying submission.

## What is intentionally NOT included

This archive contains **no experimental results** and no precomputed output
that could reveal the reported numbers. It also contains no benchmark data,
model weights, embedding caches, vector indexes, OpenIE caches, PDFs, paper
sources, Git history, repository URLs, author names, or machine-specific paths.

The following directories are intentionally empty in the archive and are
created locally by the reproduction process:

- `data/` — downloaded public benchmark assets;
- `external/` — pinned upstream repositories;
- `artifacts/` — locally generated indexes, caches, retrieval traces and metrics.

## Reproducibility scope

The package contains the core implementation and transparent protocol drivers
for the paper's main experimental stack:

1. dense passage retrieval;
2. OpenIE graph construction and fact indexing;
3. recognition-guided PPR retrieval;
4. BridgeRAG-compatible bridge-conditioned candidate construction/scoring;
5. non-evicting LON graph expansion;
6. fixed-pool LON-RAG-L Lorentzian ordering;
7. paired bootstrap statistics;
8. matched evaluation on MuSiQue, 2WikiMultiHopQA and HotpotQA;
9. no-target-tuning transfer to GraphRAG-Bench Novel.

The artifact does **not** ship model outputs. Reproduction therefore performs
all LLM calls and embedding/index construction from scratch.

---

## 1. Environment

Recommended environment:

- Python >= 3.10;
- Linux or macOS;
- enough disk space for downloaded public corpora, Hugging Face caches and
  locally generated indexes;
- a CUDA-capable GPU is strongly recommended for the exact large-model setup;
- Ollama (or an equivalent endpoint after adapting the provider configuration).

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[benchmarks,test]'
```

The paper configuration uses:

- passage/fact embedding: `nvidia/NV-Embed-v2`;
- recognition filter: `llama3.3:70b`;
- answer reader: `llama3.3:70b`;
- bridge planning / entity extraction / conditional judging: `qwen3:30b-a3b`;
- deterministic decoding (`temperature=0`), with reasoning/thinking disabled
  where supported by the local endpoint.

For an Ollama setup:

```bash
ollama pull llama3.3:70b
ollama pull qwen3:30b-a3b
export OLLAMA_BASE_URL=http://localhost:11434
```

Useful local cache variables:

```bash
export CLM_HF_CACHE="$PWD/.clm_data/huggingface"
export CLM_EMBED_CACHE="$PWD/.clm_data/nv_embed_query_batches"
export CLM_VECTOR_BACKEND=numpy
```

Do not place caches inside the submission archive.

---

## 2. Frozen experimental rules

These rules are part of the protocol and should not be changed after inspecting
holdout results.

### Local multi-hop benchmarks

For each benchmark, questions are kept in released order after validating
corpus identity:

- first **100 valid questions**: calibration;
- final **900 valid questions**: frozen holdout.

All comparisons are paired by stable released IDs. A run must fail if IDs,
question text, or gold passage identity disagree between methods.

### Recognition width

Candidate fact widths are calibrated from:

```text
{5, 20}
```

A wider view advances only if calibration Recall@5 is **strictly greater** than
the parent/control view. Ties do not advance.

### LON graph budget

The graph-only append budget is selected on calibration from:

```text
{1, 3, 5, 10}
```

The Bridge candidate pool must be a subset of the LON pool for every query.
Graph expansion appends candidates; it never evicts a parent candidate.

### LON-RAG-L weight

The fixed-pool Lorentzian weight is selected on calibration from:

```text
{0.00, 0.05, 0.10, 0.20}
```

Ties prefer the smaller value. `gamma=0` **must reproduce the source LON
ranking exactly**. The bridge, candidate set, ANN scores and LLM judge scores
are frozen; only the bridge-relative Lorentzian order percentile is added.

### Answer generation

Answer generation is performed only after retrieval selection is frozen. The
same reader, prompt, top-five evidence budget and deterministic decoding are
used for all retrievers in a matched comparison.

### Statistics

All uncertainty intervals for method differences are paired bootstrap 95%
intervals over per-query differences, with a fixed random seed.

---

## 3. Public benchmark assets

No benchmark data is packaged here.

### HippoRAG-aligned multi-hop assets

The downloader pins the public HippoRAG source revision and checks hashes for
assets whose exact digest is frozen in the artifact.

```bash
python scripts/download_hipporag_reference_assets.py --asset prompt
python scripts/download_hipporag_reference_assets.py --asset qa-prompt
python scripts/download_hipporag_reference_assets.py --asset musique-queries
python scripts/download_hipporag_reference_assets.py --asset musique-corpus
python scripts/download_hipporag_reference_assets.py --asset 2wiki-queries
python scripts/download_hipporag_reference_assets.py --asset 2wiki-corpus
python scripts/download_hipporag_reference_assets.py --asset 2wiki-openie
python scripts/download_hipporag_reference_assets.py --asset hotpot-queries
python scripts/download_hipporag_reference_assets.py --asset hotpot-corpus
```

HotpotQA is downloaded from the same pinned public HippoRAG revision. The
downloader prints the observed SHA-256 so it can be recorded in a reproduction log.

For any dataset without a precomputed OpenIE file, the artifact includes a
fully local fallback that preserves corpus order and generates aligned OpenIE
from the same retriever-side model:

```bash
python scripts/generate_openie_from_corpus.py \
  --corpus data/hipporag2/musique_corpus.json \
  --output artifacts/musique/openie.json \
  --cache-dir .clm_data/musique_openie \
  --model qwen3:30b-a3b
```

The generated file is a local artifact and must not be added back to the
submission ZIP.

---

## 4. Build a dataset stack from scratch

Materialize one benchmark retrieval unit per source passage:

```bash
python scripts/materialize_hipporag_corpus.py \
  --corpus data/hipporag2/musique_corpus.json \
  --output artifacts/musique/docs
```

Then build the passage memory, geometry and fact index:

```bash
python scripts/prepare_openie_stack.py \
  --docs artifacts/musique/docs \
  --openie artifacts/musique/openie.json \
  --output-root artifacts/musique/stack \
  --text-encoder nv-embed-v2 \
  --text-model nvidia/NV-Embed-v2 \
  --dim 8 \
  --steps 400 \
  --seed 13 \
  --embedding-batch-size 4 \
  --training-backend torch
```

The important invariant is **one benchmark passage = one retrieval chunk** for
the local multi-hop experiments.

Repeat with separate output roots for 2WikiMultiHopQA and HotpotQA. Never reuse
an index from another corpus: the evaluation helpers verify title/text identity
and fail closed on incompatible cached indexes.

---

## 5. Recognition-guided retrieval

The code path is implemented by:

- `src/clm/clm/recognition.py`;
- `src/clm/clm/retrieval.py`;
- `scripts/evaluate_recognition.py`;
- `scripts/evaluation_common.py`.

Run calibration and the frozen split with both candidate widths:

```bash
PYTHONPATH=src/clm:scripts python scripts/evaluate_recognition.py musique \
  --data-root data/hipporag2 \
  --index artifacts/musique/stack/index \
  --fact-index artifacts/musique/stack/fact_index \
  --prompt data/hipporag2/filter_llama3.3-70B-Instruct.json \
  --cache-root .clm_data/musique_recognition \
  --candidate-counts 5 20 \
  --model llama3.3:70b \
  --output-root artifacts/musique/recognition
```

When reproducing the paper selection logic, use holdout numbers only for the
configuration admitted by the calibration rule above. The cache exists only to
make interrupted model calls resumable; it is not part of the submitted
artifact.

---

## 6. BridgeRAG-compatible parent and non-evicting LON

The exact deterministic candidate accounting is in
`scripts/bridge_paper_protocol.py`. The executable evaluator is
`scripts/evaluate_bridge_conditioning_paper.py`.

### Bridge parent

```bash
PYTHONPATH=src/clm:scripts python scripts/evaluate_bridge_conditioning_paper.py musique \
  --data-root data/hipporag2 \
  --index artifacts/musique/stack/index \
  --output artifacts/musique/bridge_cal100.json \
  --model qwen3:30b-a3b \
  --start 0 --limit 100 \
  --alpha 0.1 \
  --cache-dir .clm_data/musique_bridge
```

The parent protocol is:

1. dense top-1 bridge;
2. generate exactly three SVO next-hop queries;
3. generate exactly two bridge entities;
4. SVO ANN top-10 per query, union-max, retain SVO-15;
5. entity ANN top-5 per entity;
6. union/cap the parent candidate set at 20;
7. conditionally judge every candidate using `(query, bridge, candidate)`;
8. percentile-normalize judge and SVO cosine scores;
9. fuse with `alpha=0.1`;
10. output five passages.

### LON reachability intervention

Pass the admitted recognition artifact as a graph source and append, rather
than replace, graph candidates:

```bash
PYTHONPATH=src/clm:scripts python scripts/evaluate_bridge_conditioning_paper.py musique \
  --data-root data/hipporag2 \
  --index artifacts/musique/stack/index \
  --output artifacts/musique/lon_cal100.json \
  --model qwen3:30b-a3b \
  --start 0 --limit 100 \
  --alpha 0.1 \
  --graph-artifact artifacts/musique/recognition/recognition_top20_calibration.json \
  --graph-field passage_ids \
  --graph-top-k 5 \
  --cache-dir .clm_data/musique_bridge
```

For the paper protocol, repeat calibration for graph budgets `1 3 5 10`, select
only on calibration, and then run exactly the selected budget on the final 900
questions (`--start 100 --limit 900`).

Mechanical invariant:

```text
C_B(q,b) subseteq C_L(q,b)   for every query
```

No parent candidate may be removed by LON.

---

## 7. Fixed-pool LON-RAG-L

`scripts/evaluate_lon_lorentz_variant.py` performs no new LLM calls. It reuses
the exact LON candidate pool, bridge, ANN scores and conditional judge scores.

First use `gamma=0` as an integrity check:

```bash
PYTHONPATH=src/clm:scripts python scripts/evaluate_lon_lorentz_variant.py \
  --index artifacts/musique/stack/index \
  --source artifacts/musique/lon_cal100.json \
  --output artifacts/musique/lon_l_gamma0_cal100.json \
  --gamma 0 \
  --expected 100
```

This command must fail if the resulting ranking is not identical to LON.

Then evaluate the frozen calibration grid:

```bash
for G in 0.05 0.10 0.20; do
  PYTHONPATH=src/clm:scripts python scripts/evaluate_lon_lorentz_variant.py \
    --index artifacts/musique/stack/index \
    --source artifacts/musique/lon_cal100.json \
    --output "artifacts/musique/lon_l_gamma${G}_cal100.json" \
    --gamma "$G" --expected 100
done
```

Choose the smallest gamma with a strict calibration improvement; otherwise
retain `gamma=0`. Only after this decision is frozen should the selected gamma
be applied to the 900-query LON holdout artifact.

The directional score is:

```text
kappa(b,c) = tau(c) - tau(b) - d_H(z(b), z(c))
```

and is used only as a within-pool percentile. It is not interpreted as physical
time or ground-truth causality.

---

## 8. 2WikiMultiHopQA and HotpotQA replication

Repeat Sections 4--7 with separate data/index/output/cache roots. Keep the
following fixed across systems within each benchmark:

- exact corpus and released stable IDs;
- NV-Embed-v2 passage vectors;
- output budget `k=5`;
- same retriever-side models and deterministic decoding;
- same answer reader and prompt;
- no answer-level tuning of retrieval.

The 2Wiki public OpenIE asset can be fetched with:

```bash
python scripts/download_hipporag_reference_assets.py --asset 2wiki-openie
```

If regenerating OpenIE instead, record that provenance explicitly; do not mix a
regenerated graph with claims of byte-for-byte reproduction of a public OpenIE
artifact.

---

## 9. GraphRAG-Bench Novel transfer

This stage is a **no-target-tuning transfer test**.

Clone the official GraphRAG-Bench repository separately and check out exactly:

```text
fdbab5959b18c96532580877ffe27d112bccc0ec
```

Example layout:

```text
external/GraphRAG-Benchmark/
```

Prepare the Novel corpora with the official chunking configuration used by the
artifact: 256 tokens with 32-token overlap.

```bash
PYTHONPATH=src/clm:scripts python scripts/prepare_graphrag_bench.py \
  --benchmark-root external/GraphRAG-Benchmark \
  --subset novel \
  --output-root data/graphrag_bench \
  --tokenizer nvidia/NV-Embed-v2 \
  --chunk-tokens 256 \
  --overlap-tokens 32
```

For each prepared corpus directory, build a fresh stack:

```bash
PYTHONPATH=src/clm:scripts python scripts/build_graphrag_bench_stack.py \
  --corpus-dir data/graphrag_bench/novel/<CORPUS> \
  --output-root artifacts/graphrag_bench/novel/<CORPUS>/stack \
  --openie-model qwen3:30b-a3b \
  --embedding-model nvidia/NV-Embed-v2 \
  --geometry-steps 400 \
  --geometry-dim 8 \
  --seed 13
```

Run both question types with settings inherited from the MuSiQue-frozen
pipeline:

```text
recognition fact candidates = 20
graph append g             = 5
alpha                      = 0.1
gamma                      = 0.2
recognition model          = llama3.3:70b
bridge/LON model           = qwen3:30b-a3b
embedding                  = nvidia/NV-Embed-v2
```

Example:

```bash
PYTHONPATH=src/clm:scripts python scripts/evaluate_graphrag_bench_corpus.py \
  --corpus-dir data/graphrag_bench/novel/<CORPUS> \
  --stack-root artifacts/graphrag_bench/novel/<CORPUS>/stack \
  --recognition-prompt data/hipporag2/filter_llama3.3-70B-Instruct.json \
  --question-type 'Complex Reasoning' \
  --output artifacts/graphrag_bench/novel/<CORPUS>/complex_reasoning.json
```

Then repeat with:

```bash
--question-type 'Fact Retrieval'
```

**Do not use GraphRAG-Bench labels to select recognition width, graph budget,
alpha or gamma.** The transfer result is invalid if these are retuned on the
target benchmark.

---

## 10. Answer-level evaluation

After retrieval is frozen, generate answers for each retrieval condition using
the same `llama3.3:70b` reader, the same QA prompt, the same top-five evidence
budget and deterministic decoding. Keep answer artifacts separate from
retrieval selection so generation cannot influence calibration.

Use exact match and token-level F1. Pair answer comparisons by stable released
IDs. Any parse failure should be retained and counted rather than silently
removed; complete-case sensitivity can be reported as a secondary audit.

The QA prompt is downloaded by:

```bash
python scripts/download_hipporag_reference_assets.py --asset qa-prompt
```

---

## 11. Required integrity checks

Before computing final tables, verify all of the following:

```text
[ ] calibration IDs and holdout IDs are disjoint
[ ] every paired method has identical IDs, questions and gold evidence
[ ] Bridge candidate pool is a subset of LON candidate pool query-by-query
[ ] LON and LON-RAG-L have identical candidate exposure
[ ] gamma=0 reproduces the source LON ranking exactly
[ ] no answer metric was used for retrieval calibration
[ ] GraphRAG-Bench uses the frozen source-benchmark hyperparameters
[ ] incomplete or partially written JSON artifacts are rejected
[ ] parse/model failures remain visible in provenance
```

Run the lightweight package checks:

```bash
python -m compileall -q src/clm scripts
pytest -q
```

---

## 12. Recreating tables without shipping results

This ZIP deliberately contains no `generated_*results*`, paper macros, or
numerical output. The normal workflow is:

```text
public data/model download
        -> local OpenIE/index construction
        -> calibration artifacts
        -> frozen holdout artifacts
        -> answer artifacts
        -> local aggregation/statistics
        -> paper tables/macros
```

Keep the final four stages outside this ZIP. The artifact is designed so a
reviewer starts from code and public inputs rather than from precomputed paper
numbers.

---

## 13. Expected computational cost

The archive itself is intentionally small. The expensive resources are created
only during reproduction:

- large LLM weights and NV-Embed-v2 weights;
- Hugging Face/Ollama caches;
- OpenIE responses;
- passage/fact indexes;
- per-query LLM judge/answer caches;
- GraphRAG-Bench per-corpus stacks.

These may require many GB of local storage and substantial GPU time. None is
part of the submitted archive.

## 14. Anonymity

This package intentionally omits author identities, private repository links,
Git metadata, local usernames, institution names and machine-specific absolute
paths. Please report any accidental identity-bearing metadata through the
conference artifact channel rather than attempting to identify the authors.

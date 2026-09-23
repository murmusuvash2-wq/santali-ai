# Santali AI application and API

The repository's research pipeline is now paired with a WebDev full-stack application named **Santali AI**. The application is intentionally a research preview: it exposes translation and assistant contracts while clearly reporting when the real model gateway or grounded retrieval is not ready.

## Current application surface

The public website presents the product direction, the Ol Chiki-first design principles, and the current corpus and release-stage facts. The workspace exposes translation between `eng_Latn`, `sat_Olck`, and `hin_Deva`, plus a safety-aware assistant preview. The API docs page documents the typed procedures used by the frontend.

The backend procedures are:

| Procedure | Purpose |
|---|---|
| `app.health` | Report service, model-gateway, and retrieval readiness. |
| `app.stats` | Report the public corpus and research-preview stage. |
| `app.supportedLanguages` | Return supported language and script metadata. |
| `app.translate` | Translate text through the server-side model gateway or an explicit preview fallback. |
| `app.ask` | Answer a question through the server-side model gateway or a qualified preview fallback. |
| `app.history` | Return recent interaction history for a signed-in user. |
| `app.apiCatalog` | Describe the public procedure contract for downstream clients. |

## What remains before a production claim

The application does not claim that IndicTrans2 has been fine-tuned. The research pipeline must still produce a baseline, pass tokenizer and numerical preflight, complete a reproducible LoRA run, report BLEU/spBLEU and chrF++, and receive native Santali review. The knowledge assistant also needs an approved, citation-preserving retrieval corpus before it should answer changing public-service, agriculture, medical, or legal questions.

## Local app development

The WebDev project is created separately from this research repository so runtime, authentication, database, and deployment concerns remain isolated from large model and dataset artifacts. Use the repository's `scripts/prepare_experiment.py` to create a traceable data package before any training run.

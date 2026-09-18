# Non-commercial Santali AI permission plan

**Status:** non-commercial research only  
**Version:** 1.0  
**Reviewed:** 2026-09-19

## Principle

Non-commercial does not mean permission-free. Every training source must have a verified license or written permission, an exact version/checksum, an attribution record, and a quality review. Unknown-license material remains excluded.

## Data lanes

| Lane | Use | Release rule |
|---|---|---|
| Research | Non-commercial training and private Kaggle experiments | No public dataset/model redistribution unless separately permitted |
| Redistributable | Public-domain, CC0, or explicitly cleared material | License notices and attribution must ship with every release |
| Evaluation-only | Benchmark or restricted data | Never used for training |

CC BY-NC-SA sources may be used only in the research lane unless the rights holder grants additional permission. They must not be mixed into a redistributable or commercial checkpoint by default.

## Permission scope

Written permission should clearly cover:

1. Machine-learning training for non-commercial research.
2. Cleaning, Unicode normalization, deduplication, and other transformations.
3. Private Kaggle/research hosting, when applicable.
4. Publishing aggregate metrics and selected examples.
5. Dataset, adapter, or model redistribution, if intended.
6. Required author credit, source URL, license, and change notice.
7. For audio: speaker consent, transcription, retention, and hosting.
8. A correction or removal contact.

Keep original permission messages privately. Public manifests should contain only a redacted evidence ID such as `perm-0001`.

## Candidate sources to review

- [sami42200/santali-nlp](https://github.com/sami42200/santali-nlp): inspect `parallel_en_sat.csv`; preserve upstream terms and checksum.
- [Prasanta-Hembram translation work-list](https://github.com/Prasanta-Hembram/Translation-work-list-for-Santali-language): request file-level permission for dictionaries and linked resources; do not assume linked sources share the repository license.
- [samarsoren spell-suggestion wordlist](https://github.com/samarsoren/santali-spell-suggestion-wordlist): treat as research-only under CC BY-NC-SA 4.0 until additional permission is obtained.
- [Tatoeba downloads](https://tatoeba.org/en/downloads): preserve sentence IDs, contributor names, and each sentence’s displayed license.
- Direct native contributions: prefer original sentences and translations submitted under a project-approved license.

Do not train on scraped articles, undocumented Drive files, unknown-license crawlers, or audio without consent.

## Provenance record

Use one record per artifact:

```yaml
source_id: example-source
name: Human-readable name
url: https://example.org
exact_version: commit-or-release
sha256: hex-digest
license: SPDX-or-human-readable-license
license_url: https://example.org/license
rights_holder: name
permission_evidence: perm-0001
training_use: noncommercial_research_only
redistribution: prohibited_until_cleared
attribution: Preferred credit text
changes: Unicode normalization; empty-row removal; exact-pair deduplication
review_status: unreviewed
removal_contact: contact-or-evidence-id
```

Do not commit private emails, consent forms, recordings, or identity documents to GitHub. Keep only redacted evidence IDs in public manifests.

## Credit standard

Every dataset version, Kaggle description, model/adapter card, and dashboard About page must credit permitted sources. For Tatoeba, include Tatoeba, sentence IDs, contributor usernames, and sentence-specific licenses. For CC BY-SA material, include the license, attribution, and ShareAlike notice. Credit is a release requirement.

Example:

```markdown
## Data attribution
- MMLoSo 2025 English–Santali shared task — license and source URL; changes are listed in the manifest.
- Tatoeba — sentence IDs and contributor names are listed in the manifest.
- Community contributors — credited under the permission evidence recorded in the private ledger.
```

## Review workflow

A source passes only after both rights and quality review:

```text
unreviewed -> community_reviewed -> native_reviewed -> approved
```

Review language direction, Ol Chiki accuracy, Latin transliteration, Hindi/Bengali code-switching, spelling, fluency, named entities, offensive content, and personal information. Keep submitted text and reviewer corrections as separate versions.

## Release gates

Before a private research run, record permission, URL, version, checksum, attribution, and privacy/rehosting conditions. Before a redistributable package, confirm redistribution is explicitly allowed and all notices can be included.

Run the repository gate before every package build:

```bash
python scripts/audit_corpus.py --output-dir .corpus-audit
```

Archive the audit output, source manifest, and generated `ATTRIBUTION.md` with the exact Kaggle dataset version. If permission is withdrawn, quarantine the source, identify dependent checkpoints, and stop new training until the issue is resolved.

## Permission request template

> Namaskar. We are building a non-commercial open research project for Santali and Ol Chiki language technology at [project URL]. We would like to use [exact file/sentences/recordings] from [source URL and commit/release] for language research and model training. We will preserve your name, source link, license, and required notices. We will not use the material commercially or redistribute it beyond the permissions you grant. May we normalize and deduplicate it, upload it to a private Kaggle research dataset, and publish aggregate evaluation results? Please state restrictions, preferred credit text, removal contact, and whether dataset or model redistribution is allowed.

## First milestone

Build a **10,000–20,000 pair Ol Chiki research seed** with complete provenance and native review. Keep MMLoSo, Tatoeba, GitHub dictionaries, and community contributions in separate partitions so that a rights or quality issue can be removed without invalidating the entire dataset.

Describe the project as a non-commercial research prototype until licensing, base-model terms, evaluation quality, and release rights are reviewed. This document is engineering intake policy, not legal advice; when ownership or terms are unclear, pause and ask the rights holder.

## Maintainer checklist

- [ ] License or permission evidence archived.
- [ ] Exact source version and SHA-256 recorded.
- [ ] Training and derivative scope confirmed.
- [ ] Private Kaggle hosting checked.
- [ ] Attribution text approved.
- [ ] Native review sample completed.
- [ ] PII and speaker-consent checks completed.
- [ ] Research or redistributable lane assigned.
- [ ] Audit output archived with the dataset version.

**Operational rule:** a smaller licensed, credited, native-reviewed, removable corpus is better than a larger unverified corpus.

The source registry and `scripts/audit_corpus.py` remain the final eligibility gate.

## References

- [Tatoeba terms of use](https://tatoeba.org/en/terms_of_use)
- [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- [Tatoeba downloads](https://tatoeba.org/en/downloads)

**Maintainer:** Santali AI contributors

**Credit policy:** required for every permitted source.

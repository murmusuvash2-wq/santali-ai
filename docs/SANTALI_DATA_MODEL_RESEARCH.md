# Santali data and model research

## Executive conclusion

For Santali AI, the safest primary stack is **open IndicTrans2 plus independent benchmarking with NLLB-200/FLORES**, trained only on rights-cleared Santali data. Google Translate should be treated as a black-box comparison system or weak-label teacher only after its terms are checked. Microsoft Azure Translator should not be treated as a current Santali translation dependency because Santali was not listed in the reviewed Azure Translator language table, although Microsoft Azure Language Detection lists `sat` with the Ol Chiki script code `Olck`.[1] [3]

Google's consumer Translate interface lists **Santali (Latin)** and **Santali (Ol Chiki)**, but this does not mean that Google Cloud Translation exposes the same capability. The reviewed Google Cloud language list did not show Santali in its translation tables.[3] [5] Therefore, consumer Google support, Google Cloud API support, and a downloadable Google model must be treated as three separate questions.

The most practical open starting point is **AI4Bharat IndicTrans2**. Its official inventory includes Santali as `sat_Olck`, Hindi as `hin_Deva`, Bengali as `ben_Beng`, and English as `eng_Latn`. This makes it suitable for direct Ol Chiki experiments across these languages, but the inventory does not establish a native Roman-Santali code such as `sat_Latn`.[7]

## Microsoft versus Google versus open models

| Option | What is verified | What is not verified | Recommended use |
|---|---|---|---|
| Microsoft Azure Translator | Hindi and Bengali are listed for translation; Santali appears in Azure language detection with Ol Chiki script metadata | Santali translation endpoint, Roman Santali support, quality, and custom Santali training | Verify availability with Microsoft before considering it; do not make it a dependency now |
| Google Translate consumer | Santali Latin and Santali Ol Chiki are listed in the consumer product | Cloud API support, downloadable weights, quality by direction/script, and training rights for outputs | Use as a comparison/oracle only after terms and privacy are checked |
| Google Cloud Translation | Glossary support is documented, but Santali was not found in the reviewed translation language list | Santali API availability and custom model support | Use for supported Hindi/Bengali terminology workflows, not as confirmed Santali infrastructure |
| AI4Bharat IndicTrans2 | Explicit `sat_Olck` support and downloadable checkpoints/scripts | Strong Roman Santali capability and broad code-switching quality | Primary open baseline and fine-tuning target |
| NLLB-200/FLORES-200 | Santali `sat_Olck` is included in the multilingual inventory | Product-quality ranking for Santali and all intended licenses for redistribution | Independent baseline and held-out evaluation anchor |

## What public data is realistically available?

### MMLoSo Santali parallel data

The clearest recent Santali translation resource is the MMLoSo 2025 English–Santali task. The official paper describes a **20,000-pair Santali training arm** with Ol Chiki targets and a 15,999-sentence shared-task test inventory. A later study reports expanding its training pool with public resources to 107,975 parallel sentences, but that larger number should not be treated as an official single-source release without checking the exact files and licenses.[11] [12]

The paper reports that the data were sentence-aligned, normalized, deduplicated, and filtered. It also states that the released data are CC BY-SA 4.0. Any Kaggle package must retain attribution, license text, and share-alike obligations where applicable.

### AI4Bharat BPCC and IndicTrans2 artifacts

AI4Bharat provides the strongest ecosystem for experimentation: models, training scripts, parallel-data references, back-translation artifacts, and evaluation tooling. BPCC supports Santali, but the published corpus-wide total must not be presented as a Santali-specific count. The Santali slice has to be downloaded, filtered, counted, and audited at the component level.[7]

The repository describes mixed artifact licensing. Model checkpoints are documented under MIT. Some data artifacts are described as CC0, while BPCC-H-Wiki, BPCC-H-Daily, and IN22 are described as CC BY 4.0. Packaging status does not automatically clear every underlying extracted web sentence. The source, exact revision, license notice, attribution, and SHA-256 should travel with every Kaggle artifact.

### Evaluation resources

FLORES-200 and IN22 should be kept as held-out evaluation resources rather than mixed into training. IndicGenBench explicitly warns against using its data in pretraining. The test set must remain separate so that improvements are measured on sentences the model has not seen.[9] [10]

### Hindi and Bengali code-switch data

Open Hindi-English and Bengali-English code-switch datasets exist, including GLUECoS and OpenSLR speech resources. These are useful for building code-switch detection, transliteration, and routing components, but they are **not Santali data**. Roman Hindi or Bengali cannot be silently relabelled as Roman Santali. Any Hindi/Bengali examples included in the Santali system must keep separate language, script, and source fields.[13] [14]

### Roman or Latin Santali

Public Roman-Santali data is scarce. The Sanlish model card describes a 502-recording thesis dataset split into 401 training, 50 validation, and 51 test recordings, but explicitly says the underlying dataset is not publicly released. The model license does not license those recordings or annotations.[15]

A practical Roman-Santali component will therefore require permission-based collection from native speakers. The transliteration convention must be recorded because Roman Santali is not a single standardized spelling system. The original user spelling should be preserved alongside any normalized or Ol Chiki representation.

### Speech and transcription

AI4Bharat publishes a downloadable Santali IndicConformer model under its open model ecosystem. A peer-reviewed ASR study evaluated Ol Chiki Santali speech and found that Bengali-pretrained Whisper Small reached 28.47% WER on its reported Common Voice setup, while Hindi-pretrained Whisper Small reached 34.50%. These are narrow study results, not general accuracy claims for every dialect, script, or code-switch pattern.[16]

Common Voice Santali is useful for an ASR pilot, but it is small and its exact current release must be verified. Its dataset terms and re-hosting limitations should be checked before copying raw audio into Kaggle. A safer first step is a controlled local workflow with a manifest and speaker-independent splits.

## Can Google or Microsoft quality be transferred into our model?

**Partially in principle, but not automatically and not yet proven for this project.** A black-box teacher can generate translations for carefully selected inputs, and a student may learn terminology or style patterns from those outputs. However, proprietary APIs do not expose weights, logits, alternative translations, alignment information, or the teacher's uncertainty.

The legal boundary is equally important. Before storing API outputs as training data, verify the provider's current terms for model training, retention, commercial use, redistribution, PII, rate limits, and derived models. Do not bulk-scrape a consumer interface. Do not claim that a resulting model is a Google- or Microsoft-distilled model without documented permission.

If permitted, run a small, labelled A/B experiment:

1. Use the same licensed source sentences for the open baseline and the teacher.
2. Store provider, date, product/model version, direction, script, prompt, and permission record for every synthetic row.
3. Have native Santali reviewers score a stratified sample.
4. Compare the open baseline, human-data fine-tune, teacher pseudo-labels, and quality-filtered pseudo-labels on an untouched community-reviewed test set.
5. Keep provider outputs out of the production training run until the permission and data-retention questions are resolved.

## Recommended benchmark plan

Freeze separate test sets for English↔Ol Chiki, Hindi↔Santali, Bengali↔Santali, Ol Chiki↔Latin, and genuine code-switching. Track script and direction explicitly. Report BLEU, chrF++, spBLEU, and where practical COMET. Break results down by sentence length, domain, named entities, numbers, script, and code-switch rate.

The benchmark should include:

- IndicTrans2 direct translation.
- NLLB-200 as an independent open baseline.
- Google consumer output where it can be collected under applicable terms.
- Microsoft only if a documented Santali translation endpoint is confirmed.
- A pivot baseline such as Santali→English→Hindi for diagnosis, not as a production claim.
- Native-speaker ratings for adequacy, fluency, Ol Chiki spelling, omissions, additions, names, numbers, and naturalness.

A single aggregate score will hide the important direction gap. The published English–Santali results show this clearly: one 2025 study reports 7.3 BLEU/40.3 chrF++ for English→Santali and 26.8 BLEU/53.9 chrF++ for Santali→English on IN22-Gen; its FLORES results are 4.7/32.7 and 22.0/49.2 respectively.[11] These are research-model results, not Google or Microsoft scores.

## Recommended next steps for Santali AI

First, add MMLoSo as a separately attributed, CC BY-SA 4.0 component after exact-release verification. Second, extract and audit the Santali slice of BPCC rather than importing the corpus-wide total. Third, use IndicTrans2 and NLLB-200 as reproducible baselines. Fourth, build a permission-based Roman/Sanlish contribution set instead of relying on an unreleased thesis corpus. Fifth, keep Hindi/Bengali code-switch data in separate buckets and use them for routing or transfer learning only after measuring their effect on Santali.

The current repository license gate is the correct foundation. It should approve a source only after exact file paths, release/commit, training permission, redistribution/private-hosting permission, attribution, and SHA-256 are recorded. The current 63,179-pair Mod4 source should remain excluded until its upstream license is confirmed.

## References

[1]: https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support "Microsoft Azure Translator language support"
[2]: https://learn.microsoft.com/en-us/azure/ai-services/translator/custom-translator/concepts/model-training "Microsoft Custom Translator model training"
[3]: https://docs.cloud.google.com/translate/docs/languages "Google Cloud Translation supported languages"
[4]: https://docs.cloud.google.com/translate/docs/advanced/glossary "Google Cloud Translation glossary"
[5]: https://translate.google.com/ "Google Translate consumer interface"
[6]: https://indianexpress.com/article/technology/tech-news-technology/google-translate-expands-to-110-new-languages-including-marwadi-santali-and-tulu-9420834/ "Report on Google's 2024 language expansion"
[7]: https://github.com/AI4Bharat/IndicTrans2 "AI4Bharat IndicTrans2 repository"
[8]: https://indicnlp.ai4bharat.org/samanantar/ "AI4Bharat Samanantar"
[9]: https://github.com/facebookresearch/flores/blob/main/flores200/README.md "FLORES-200 README"
[10]: https://huggingface.co/datasets/google/IndicGenBench_flores_in "IndicGenBench FLORES-IN dataset card"
[11]: https://aclanthology.org/2025.mmloso-1.9/ "English-Santali translation study"
[12]: https://aclanthology.org/2025.mmloso-1.14/ "MMLoSo Santali data and shared-task paper"
[13]: https://github.com/microsoft/gluecos "GLUECoS code-switched corpus"
[14]: https://www.openslr.org/104/ "OpenSLR Hindi-English and Bengali-English speech corpus"
[15]: https://huggingface.co/thunderboltc/whisper-small-santali-sanlish "Sanlish model card"
[16]: https://aclanthology.org/2025.findings-ijcnlp.16/ "Santali ASR and multilingual transfer study"

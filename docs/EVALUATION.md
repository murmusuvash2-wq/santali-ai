# Evaluation plan

## Translation

Report the existing-model baseline and fine-tuned result on the same hidden test set. Use BLEU and chrF++ for comparison, then add native-speaker scores for meaning, grammar, naturalness and Ol Chiki correctness. Do not report only an internal score; include an unseen domain split.

## Chatbot

Evaluate groundedness, answer helpfulness, Santali fluency, refusal behavior, source citation and code-switching. Include questions that are not present in the knowledge base and verify that the system says it does not know.

## Speech

For ASR report character error rate and word error rate by speaker, region, noise level and script. For TTS report intelligibility, pronunciation, naturalness and speaker consistency from native listeners.

## Safety

Test medical emergencies, harmful instructions, hate speech, privacy requests, political misinformation, financial/legal claims and ambiguous questions. Health and emergency content must route to official services or a qualified human rather than inventing advice.

## Release gates

- Reproducible preprocessing.
- No train/test leakage.
- Dataset and model cards complete.
- Two native-speaker reviewers approve the hidden sample.
- Baseline comparison included.
- Source-grounded RAG answers.
- Security and rate limits configured before public API exposure.

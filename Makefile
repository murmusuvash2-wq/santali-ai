.PHONY: check diagram audit-corpus profile-parallel import-mmloso build-seed-corpus prepare-experiment evaluate-predictions test
check:
	python -m compileall scripts training src

test:
	python -m unittest discover -s tests -p 'test_*.py'

diagram:
	manus-render-diagram diagrams/architecture.mmd diagrams/architecture.png

audit-corpus:
	python scripts/audit_corpus.py --output-dir .corpus-audit

profile-parallel:
	@test -n "$(INPUT)" -a -n "$(OUTPUT)"
	python scripts/profile_parallel.py --input "$(INPUT)" --output "$(OUTPUT)"

import-mmloso:
	@test -n "$(INPUT)" -a -n "$(SHA256)" -a -n "$(OUTPUT)"
	python scripts/import_mmloso.py --input "$(INPUT)" --source-sha256 "$(SHA256)" --output "$(OUTPUT)"

build-seed-corpus:
	@test -n "$(INPUT)" -a -n "$(OUTPUT_DIR)" -a -n "$(MANIFEST)"
	python scripts/build_seed_corpus.py --input "$(INPUT)" --output-dir "$(OUTPUT_DIR)" --manifest "$(MANIFEST)"

prepare-experiment:
	@test -n "$(INPUT)" -a -n "$(OUTPUT_DIR)" -a -n "$(MANIFEST)"
	python scripts/prepare_experiment.py --input "$(INPUT)" --output-dir "$(OUTPUT_DIR)" --manifest "$(MANIFEST)"

evaluate-predictions:
	@test -n "$(REFERENCES)" -a -n "$(PREDICTIONS)" -a -n "$(TRACK)" -a -n "$(OUTPUT)"
	python scripts/evaluate_predictions.py --references "$(REFERENCES)" --predictions "$(PREDICTIONS)" --track "$(TRACK)" --output "$(OUTPUT)"

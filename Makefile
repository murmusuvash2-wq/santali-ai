.PHONY: check diagram audit-corpus profile-parallel import-mmloso evaluate-predictions
check:
	python -m compileall scripts training src

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

evaluate-predictions:
	@test -n "$(REFERENCES)" -a -n "$(PREDICTIONS)" -a -n "$(TRACK)" -a -n "$(OUTPUT)"
	python scripts/evaluate_predictions.py --references "$(REFERENCES)" --predictions "$(PREDICTIONS)" --track "$(TRACK)" --output "$(OUTPUT)"

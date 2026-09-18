.PHONY: check diagram audit-corpus evaluate-predictions
check:
	python -m compileall scripts training src

diagram:
	manus-render-diagram diagrams/architecture.mmd diagrams/architecture.png

audit-corpus:
	python scripts/audit_corpus.py --output-dir .corpus-audit

evaluate-predictions:
	@test -n "$(REFERENCES)" -a -n "$(PREDICTIONS)" -a -n "$(TRACK)" -a -n "$(OUTPUT)"
	python scripts/evaluate_predictions.py --references "$(REFERENCES)" --predictions "$(PREDICTIONS)" --track "$(TRACK)" --output "$(OUTPUT)"

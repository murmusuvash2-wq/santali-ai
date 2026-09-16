.PHONY: check diagram
check:
	python -m compileall scripts training src

diagram:
	manus-render-diagram diagrams/architecture.mmd diagrams/architecture.png

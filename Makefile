.PHONY: docs

docs:
	@pdoc --output-directory docs_html salk_toolkit !salk_toolkit.tools.explorer

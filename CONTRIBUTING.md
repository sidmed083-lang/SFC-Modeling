# Contributing

1. Fork and clone, then `pip install -e ".[dashboard,dev]"`.
2. Create a branch, make focused changes, add tests. Accounting-related changes need a test that
   shows both a passing and a *failing* (deliberately broken) case.
3. Before opening a pull request run:
   ```bash
   black .
   ruff check .
   pytest -q
   python examples/validate_examples.py
   ```
4. Document equations, parameters and sign conventions of any new model; add a benchmark with a
   known solution where one exists.
5. Never commit credentials, private data or large result files.

Bug reports should include the model JSON (`spec.to_json()`), the scenario and the output of
`result.validate().summary()`.

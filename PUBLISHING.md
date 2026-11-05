# Publishing to GitHub

## One-time preparation (do this before the first push)
The code was written without access to Black, Ruff or mypy, so run them once locally and commit the result:
```bash
pip install -e ".[dashboard,dev]"
black .                  # reformat (CI runs `black --check .`)
ruff check . --fix       # lint (CI runs `ruff check .`)
mypy src                 # advisory in CI; fix what you like
pytest -q
python examples/validate_examples.py
streamlit run app.py     # click through all pages once
```
Replace `<your-user>` in `README.md` and `CITATION.cff`, and put your name in `pyproject.toml`
and `LICENSE`.

## Git commands
```bash
cd SFC-Modeling
git init -b main
git add .
git status                      # confirm no secrets, no .venv, no *.egg-info
git commit -m "Initial release: SFC-Modeling 0.1.0"

# Option A - GitHub CLI
gh repo create SFC-Modeling --public --source=. --remote=origin --push

# Option B - create an empty repo named SFC-Modeling on github.com first, then:
git remote add origin git@github.com:<your-user>/SFC-Modeling.git
git push -u origin main

git tag v0.1.0 && git push origin v0.1.0
```

## Publication checklist
- [ ] `black .`, `ruff check .`, `pytest -q`, `python examples/validate_examples.py` all pass locally
- [ ] CI green on Python 3.10, 3.11, 3.12 (first run may reveal version-specific issues)
- [ ] Dashboard clicked through for SIM, PC and BOND-REVAL (all nine pages, all downloads)
- [ ] Placeholders replaced (`<your-user>`, author names)
- [ ] No credentials or private data in the history (`git log -p | grep -i -E "key|token|secret"`)
- [ ] Repository settings: enable branch protection on `main`, require CI; Actions permissions
      "read repository contents" (the workflow already declares `permissions: contents: read`)
- [ ] Add repository description and topics (`macroeconomics`, `stock-flow-consistent`, `python`, `streamlit`)
- [ ] Create the `v0.1.0` release from the tag and paste the changelog entry
- [ ] Optional: enable Dependabot for `pip` and `github-actions`

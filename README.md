Assignment 1 — quick run instructions

1) Create and activate virtual environment (optional — a venv has been created as `.venv_assignment1`):

```bash
python3 -m venv .venv_assignment1
source .venv_assignment1/bin/activate
```

2) Install dependencies (if not already installed):

```bash
pip install -r requirements.txt
```

3) Run a part script from the Assignment folder (ensures relative CSV path resolves):

```bash
cd "$(dirname "$0")"
.venv_assignment1/bin/python part1.py
.venv_assignment1/bin/python part2.py
.venv_assignment1/bin/python part3.py
```

Notes:
- VS Code launch configuration in `.vscode/launch.json` is set to use the venv python.
- Output files from Part 3 (plots and CSVs) are written into the same folder.

# Archelec mini-project starter

Starter repository for an ENSAE NLP mini-project on the Archelec corpus.

## Recommended subject

**Predict political affiliation from the text of electoral manifestos** using a simple and solid baseline:

- text representation: **TF-IDF**
- classifier: **Logistic Regression**
- metrics: **accuracy, macro-F1, confusion matrix**

This is a good mini-project because it is easy to explain, easy to evaluate, and fully aligned with the course instructions.

## Suggested repository structure

```text
archelec_mini_project_starter/
├── data/
│   ├── raw/                # metadata CSV downloaded from Archelec
│   ├── interim/            # temporary merged files
│   ├── processed/          # final dataset ready for modelling
│   └── external/           # cloned transcription repository
├── notebooks/             # exploration notebooks
├── outputs/               # tables, figures, metrics
├── report/                # PDF article + notes
├── scripts/               # helper shell scripts
├── src/                   # Python source code
├── .gitignore
├── requirements.txt
└── README.md
```

## 1) Create a virtual environment

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Download the data

### A. Metadata CSV

According to the course note, the metadata can be downloaded in **CSV** from the Archelec explorer.

- Open the explorer in your browser
- Export the metadata as CSV
- Save it as:

```text
data/raw/archelec_metadata.csv
```

### B. Text transcriptions

The course note says the transcriptions are stored in the GitLab repository below.
Clone it into `data/external/`:

```bash
git clone https://gitlab.teklia.com/ckermorvant/arkindex_archelec.git data/external/arkindex_archelec
```

## 3) First sanity checks

### Inspect the metadata

```bash
python src/inspect_metadata.py --csv data/raw/archelec_metadata.csv
```

### Inspect the text files

```bash
python src/find_text_files.py --root data/external/arkindex_archelec
```

These two commands will help you identify:

- the available metadata columns
- the identifier that links metadata and text files
- the best candidate target column, for example `titulaire-soutien` or `liste`

## 4) Build a first modelling dataset

Once you know the right identifier and columns, create a processed CSV with at least:

- `text`
- `label`

Save it as:

```text
data/processed/classification_dataset.csv
```

## 5) Train a strong baseline

```bash
python src/train_tfidf_baseline.py \
  --csv data/processed/classification_dataset.csv \
  --text-col text \
  --label-col label
```

The script will print:

- dataset size
- label distribution
- accuracy
- macro-F1
- classification report

It will also save:

- `outputs/metrics.txt`
- `outputs/confusion_matrix.csv`

## 6) Minimal analysis to include in the report

- Description of the corpus and selected subset
- Distribution of labels
- Examples of noisy OCR or preprocessing issues
- Baseline model and justification
- Main metrics
- Confusion matrix analysis
- Short discussion of errors

## 7) Git repository setup

### Local git init

```bash
git init
git add .
git commit -m "Initial project structure"
```

### Add a GitHub remote manually

Create an empty repository on GitHub, then run:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

### Or with GitHub CLI

```bash
gh repo create archelec-mini-project --public --source=. --remote=origin --push
```

## 8) Suggested milestones

### Milestone 1
- Download the metadata CSV
- Clone the transcription repository
- Inspect columns and filenames

### Milestone 2
- Build a small clean subset
- Define 3 to 5 political labels
- Train the baseline

### Milestone 3
- Analyze confusion matrix
- Write the report
- Export the final PDF to `report/`

## 9) Important note

The course instructions say that **both the code and the PDF version of the article should be stored in a GitHub repository**, and that the article should follow the **NeurIPS 2024 layout**.

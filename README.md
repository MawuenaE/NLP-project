# NLP Project — Political Text Classification and NER on ARCHELEC OCR Corpus

## Overview

This project studies French political campaign texts extracted from the **ARCHELEC** corpus, focusing on the first round of the **1993 French legislative elections**.

The main objective is to build a complete NLP pipeline that transforms raw OCR files into an interpretable political text classification dataset, then compares classical machine learning models, deep learning models, and named entity recognition analyses.

The project follows four major stages:

1. **Corpus preparation**: extract and merge OCR texts with metadata.
2. **Labelling methodology**: construct `left` / `right` labels from political metadata.
3. **Text classification**: evaluate Bag-of-Words, TF-IDF, classical models, LSTM and GRU.
4. **NER analysis**: extract and post-process named entities to enrich the political interpretation of the corpus.



## Project Structure

```text
NLP-project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_baseline_tfidf.ipynb
│   ├── 02_deep_learning_pytorch_lstm_gru.ipynb
│   ├── 03_ner_analysis_spacy.ipynb
│
├
│
├── report/
│
├── scripts/
│
├
│
├── requirements.txt
├── README.md
└── .gitignore
```



## Folder Description

### `data/`

This folder contains the project datasets at different stages of processing.

#### `data/raw/`

Contains raw or near-raw data files.

Typical content:

```text
data/raw/
├── zips/
│   └── legislatives_1993.zip
├── pf_1993_tour1_all/
└── pf_1993_tour1_matched/
```

- `zips/`: stores downloaded ZIP archives from the external source.
- `pf_1993_tour1_all/`: contains all extracted OCR files for the 1993 legislative elections.
- `pf_1993_tour1_matched/`: contains only OCR files that match the selected metadata subset.


#### `data/interim/`

Contains intermediate datasets generated during processing.

Typical content:

```text
data/interim/
├── metadata_1993_tour1_filtered.csv
├── pf_1993_tour1_manifest.csv
├── dataset_1993_tour1.csv
├── dataset_1993_tour1_labeled.csv
└── dataset_1993_tour1_binary.csv
```

- `metadata_1993_tour1_filtered.csv`: metadata filtered to the selected subset: 1993, legislative election, first round.
- `pf_1993_tour1_manifest.csv`: manifest linking OCR files to their metadata identifiers.
- `dataset_1993_tour1.csv`: merged dataset containing metadata and OCR text.
- `dataset_1993_tour1_labeled.csv`: dataset enriched with `left`, `right`, or `other` labels.
- `dataset_1993_tour1_binary.csv`: intermediate binary dataset keeping only `left` and `right` observations.

These files document the transformation path from raw OCR to modelling data.


#### `data/processed/`

Contains the final datasets used for modelling.

Typical content:

```text
data/processed/
├── dataset_1993_tour1_binary_minimal.csv
└── dataset_1993_final_clean.csv
```

- `dataset_1993_tour1_binary_minimal.csv`: minimal binary dataset with selected columns such as `id`, `text`, and `label`.
- `dataset_1993_final_clean.csv`: final cleaned dataset used in the baseline, deep learning, and NER notebooks.

This is the main dataset used for the experiments.

---

#### `data/external/`

Contains external reference datasets used to support or validate the political mapping.

Typical content:

```text
data/external/
└── parlgov/
    └── party.csv
```

- `parlgov/party.csv`: external political party reference data used to support the mapping between party names and political orientation.



### `notebooks/`

This folder contains the main experimental notebooks.

#### `01_baseline_tfidf.ipynb`

Main classical NLP baseline notebook.

It includes:

- dataset loading;
- exploratory analysis;
- label distribution;
- word clouds;
- Bag-of-Words vectorization;
- TF-IDF vectorization;
- comparison of classical models:
  - Logistic Regression;
  - Naive Bayes;
  - Linear SVM;
- confusion matrix;
- feature importance analysis;
- bias analysis;
- debiasing experiment.

Main conclusion:

> Linear models, especially SVM and Logistic Regression, perform extremely well on this corpus. Even after removing explicit political markers, the performance remains high, suggesting that the model captures deeper lexical and discursive structures.

---

#### `02_deep_learning_pytorch_lstm_gru.ipynb`

Deep learning notebook implemented fully in **PyTorch**.

It includes:

- controlled removal of explicit political markers and top discriminative words;
- tokenization and vocabulary construction;
- PyTorch Dataset and DataLoader;
- learned input embeddings;
- LSTM classifier;
- GRU classifier;
- training and validation loops;
- early stopping;
- confusion matrices;
- comparison with the classical SVM baseline.

Observed results:

- LSTM performs better than GRU;
- both deep learning models remain below the classical SVM baseline;
- this suggests that classical sparse lexical models are highly effective for this structured corpus.

---

#### `03_ner_analysis_spacy.ipynb`

Named Entity Recognition analysis notebook using **spaCy**.

It includes:

- loading a French spaCy model;
- extracting named entities from OCR texts;
- analysing entity types:
  - `PER`;
  - `ORG`;
  - `LOC`;
  - `MISC`;
- comparing entity distributions between `left` and `right`;
- identifying the most frequent entities globally and by class;
- qualitative inspection of entity extraction.

Main purpose:

> This notebook enriches the classification task with a linguistic analysis of named entities, showing which persons, places, organizations, and institutions structure the political discourse.




### `scripts/`

This folder contains reproducible scripts used to build the dataset.

Typical scripts include:

```text
scripts/
├── prepare_1993_pf_dataset.py
├── merge_ocr_with_metadata.py
├── apply_political_mapping.py
├── build_minimal_binary_dataset.py
└── build_final_dataset_with_parlgov.py
```

Depending on the current state of the project, file names may differ slightly.

#### `prepare_1993_pf_dataset.py`

Prepares the OCR corpus for the selected election subset.

Main tasks:

- locate or load the 1993 legislative OCR archive;
- unzip the raw OCR files;
- filter metadata for:
  - year 1993;
  - legislative elections;
  - first round;
- retain OCR files matching the metadata;
- save a manifest linking OCR files and metadata.

Expected output:

```text
data/interim/metadata_1993_tour1_filtered.csv
data/interim/pf_1993_tour1_manifest.csv
data/raw/pf_1993_tour1_matched/
```

#### `merge_ocr_with_metadata.py`

Merges OCR texts with their metadata.

Main tasks:

- read metadata;
- read each OCR text file;
- join OCR text with metadata using the document ID;
- compute basic text statistics such as character count and word count.

Expected output:

```text
data/interim/dataset_1993_tour1.csv
```

#### `apply_political_mapping.py`

Applies the political orientation mapping.

Main tasks:

- load the merged OCR + metadata dataset;
- load the political mapping table;
- assign:
  - `left`;
  - `right`;
  - `other`;
- keep only binary examples when needed.

Expected output:

```text
data/interim/dataset_1993_tour1_labeled.csv
data/interim/dataset_1993_tour1_binary.csv
```

#### `build_minimal_binary_dataset.py`

Creates the minimal modelling dataset.

Main tasks:

- keep only essential columns:
  - `id`;
  - `text`;
  - `label`;
- remove duplicates and problematic cases;
- save the final dataset for modelling.

Expected output:

```text
data/processed/dataset_1993_tour1_binary_minimal.csv
```

#### `build_final_dataset_with_parlgov.py`

Uses external party reference data to support the political mapping.

Main tasks:

- load ParlGov party metadata;
- map selected French political parties to external orientation indicators when possible;
- compare or support manual mapping decisions;
- export the final clean dataset.

Expected output:

```text
data/processed/dataset_1993_final_clean.csv
```


### `report/`

Contains my final report called Elisee_Amewouame_NLP.

That report documents:

- the labelling methodology;
- the NLP classification pipeline;
- the deep learning experiments;
- the NER analysis.


## Main Results

### Classical Baseline

The baseline compared several classical NLP pipelines:

- Bag-of-Words + Logistic Regression
- Bag-of-Words + Naive Bayes
- Bag-of-Words + Linear SVM
- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes
- TF-IDF + Linear SVM

The best classical models reached approximately:

```text
Accuracy ≈ 0.99
F1-score ≈ 0.99
```

Linear SVM and Logistic Regression performed best.



### Debiasing Experiment

Because some political party names or explicit words could leak label information, a robustness experiment was performed by removing terms such as:

```text
gauche, droite, parti, socialiste, communiste, RPR, UDF, Front national
```

The model still achieved high performance after this removal, suggesting that it learned not only explicit party names, but also deeper lexical and discursive patterns.



### Deep Learning

Two PyTorch models were trained:

- LSTM
- GRU

Observed trend:

```text
SVM > LSTM > GRU
```

This indicates that deep learning from scratch did not outperform classical models on this moderate-sized, highly structured corpus.


### NER Analysis

Named Entity Recognition was used as a complementary analysis.

It showed that the corpus contains many references to:

- political parties;
- political figures;
- institutions;
- countries and territories;
- European issues such as Maastricht.

NER results were useful but noisy due to:

- OCR errors;
- abbreviations;
- political typography;
- the use of a general-purpose spaCy model.

Post-processing was therefore added to clean and normalize extracted entities.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
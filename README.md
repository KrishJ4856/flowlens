# FlowLens

**A guided network traffic lab, powered by real XGBoost and Random Forest models.**

Explore recorded network flows, edit their measurements, compare two learned detectors, and understand the results without needing an ML background. Built for a BCA minor-project presentation.

## Run the website

Use Node.js 22.13+ (tested here with Node 26) and npm:

```bash
npm ci
npm run dev
```

Open the local URL printed by Next.js. For a production static build:

```bash
npm run build
```

Next.js exports the complete site to `out/`. Deploy to Vercel by importing this repository with the Next.js preset. No environment variables, database, Python server or API keys are needed for the website. Predictions run in a Web Worker, using the actual exported fitted trees.

## What visitors can do

- **Overview:** a clear introduction and grid linking the four main sections.
- **Playground:** load a real record, edit any of 42 measurements, compare both model outputs, reveal a sample's known answer, and import/export one CSV record.
- **Training / testing data:** browse the complete partitions with pagination, whole-partition label/category filters, ID lookup and all-field record inspection. Send any record to the playground.
- **Analysis:** compare measured performance, inspect mistakes, examine train/test overlap and training-derived feature importance.
- **The project:** learn the terminology, training workflow and file structure.
- Light/dark themes, responsive layouts, keyboard-accessible controls and reduced-motion support.

## Start with these files

| Folder | Contents |
| --- | --- |
| `src/app/` | Next.js routes and page content |
| `src/components/` | Shared UI and shadcn/ui components |
| `src/lib/features.ts` | Measurement grouping and plain-language explanations |
| `public/inference.mjs` | Browser preprocessing and fitted-tree traversal |
| `public/model-worker.js` | Model download, integrity check and background inference |
| `public/models/` | Gzipped fitted-tree arrays and exact preprocessing metadata |
| `public/data/` | Paged benchmark data, samples and real analysis results |
| `ml/data/raw/` | Canonical `train.csv` and `test.csv` plus feature dictionary |
| `ml/models/` | The two original Python model bundles and saved preprocessor |
| `ml/src/`, `ml/scripts/`, `ml/config/` | Original experiment source and configuration |
| `ml/results/` | Original metrics, binary predictions, audits and frozen selection |
| `scripts/export_assets.py` | Deterministically exports the original fitted models and site data |
| `tests/` | Inference parity, input validation and complete data/filter integrity checks |
| `docs/` | Methods, report, model card, poster text, speaking notes and figures |

The original experiment also compared baselines and an attack-category extension. Its full report is retained for context; this repository bundles the **two main binary detector models**. Extra model binaries, duplicate raw source files and regenerated prepared arrays were omitted. The original full-experiment verifier expects all seven bundles; see `ml/README.md` before using those scripts.

## Real models, verified predictions

The browser does not use fake scores or a simplified surrogate. `scripts/export_assets.py` copies the fitted split thresholds and leaves from the original saved Python models into a documented binary format. Random Forest averages leaf class probabilities; XGBoost sums the saved tree outputs and applies the logistic transformation. Numeric inputs use Float32, as in the original pipeline. XGBoost accumulation also uses Float32 rounding.

**On all 82,332 frozen test records, both browser implementations produced zero Normal/Attack decision mismatches against the original Python models.** Maximum attack-score differences were 1.19e-7 for XGBoost and 3.33e-16 for Random Forest. See [`public/data/model-parity.json`](public/data/model-parity.json). These tiny numerical differences do not imply retraining.

Each binary node is 20 little-endian bytes: left child int32, right child int32, feature int32, and split threshold float64. A negative feature marks a leaf; its float64 payload holds the fitted output. Root indices, cutoffs, base margin, preprocessing and SHA-256 digests are in `public/models/manifest.json`. The browser checks model hashes before inference.

Models download only on the playground, about **6.8 MiB combined**. Inputs remain in the browser. Loading the site does not capture network traffic.

## Results to present

Measured on the complete separate test partition; settings and cutoffs were chosen using validation first.

| Model | Accuracy | Attack recall | False positive rate | Cutoff |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 87.12% | 98.12% | 26.36% | 0.50 |
| Random Forest | 88.20% | 98.28% | 24.16% | 0.55 |

XGBoost was selected using validation; Random Forest was slightly better on the final test. Around one in four normal records was falsely flagged. Scores are **uncalibrated model scores**, not verified real-world risk percentages. This is offline benchmark classification, not established live-network protection.

8,541 test records have exact training-feature matches. The report and analysis page also show the 73,791-record no-exact-match view and a unique unseen feature-plus-answer view. Training validation keeps identical feature groups together; no threshold is retuned using final test results.

## Check the implementation

```bash
npm run typecheck
npm test
npm run build
```

Tests check the complete paged data, label/category filters, source Python prediction fixtures, changed/unknown/missing inputs, input exclusion, and binary checksums.

To regenerate the website assets and repeat full-test model parity, install the recorded Linux/Python 3.11 environment first:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r ml/requirements-lock.txt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/export_assets.py
node tests/model-parity.mjs
```

This reads the saved models and canonical CSVs; it **does not train** or select new settings. Temporary full-test matrices go to ignored `.cache/model-verification/`. The website already includes the exported assets and needs no Python installation.

## Data and attribution

The dataset is **UNSW-NB15**, not a road-traffic dataset. Use the canonical filenames under `ml/data/raw/`; original mirror filenames were misleading. [`ml/data/source_manifest.json`](ml/data/source_manifest.json) records original sources, the pinned mirror revision, file hashes and filename resolution. Each raw row has 42 predictors, an ID and two answer columns. `id`, `label` and `attack_cat` are never model inputs.

- [Official UNSW-NB15 description and citation instructions](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [XGBoost paper](https://arxiv.org/abs/1603.02754)
- [Random Forests paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Next.js static exports](https://nextjs.org/docs/app/guides/static-exports)
- [shadcn/ui](https://ui.shadcn.com/), Tailwind CSS, Recharts and Lucide icons

The public benchmark and third-party libraries retain their respective attribution and usage requirements. No private chats, credentials, browsing history or environment files are included in this project.

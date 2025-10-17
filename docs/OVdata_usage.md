# Using the CPTAC Ovarian Offline Dataset

This guide explains how to generate the CPTAC ovarian cohort with `OVdata.py` and connect it to the Real-time Interactive Clinical Navigator UI.

## 1. Generate the dataset artifacts

```bash
python OVdata.py --output-dir RL0910/data/cptac_ovarian
```

The script will download the proteomics, transcriptomics, and clinical tables (only on the first run), build a multi-modal feature matrix, and export the following files inside the chosen output directory:

| File | Purpose |
| --- | --- |
| `ovarian_offline_dataset.csv` | Flat table with `patient_id`, `timestep`, `action`, `reward`, and normalized `state_*` feature columns. |
| `ovarian_offline_dataset.npz` | NumPy archive with arrays for states, actions, rewards, patient IDs, and supporting metadata. |
| `ovarian_offline_schema.yaml` | Schema mapping that allows the UI to align the CSV with the RLDT state/action format. |
| `ovarian_offline_metadata.json` | Summary statistics for reproducibility (feature catalog, normalization info, action labels, etc.). |

> **Tip:** Use `python OVdata.py --help` to adjust feature caps or disable min-max scaling if you need a different preprocessing recipe.

## 2. Load the cohort inside the UI

1. Launch `python RL0910/enhanced_chat_ui.py`.
2. In the **📊 Data Management** tab:
   - Select **Real Data** in the “Data Source” selector.
   - Upload `ovarian_offline_dataset.csv` with the “Load Real Data” file picker.
   - (Recommended) Upload `ovarian_offline_schema.yaml` in the schema slot so the UI can apply the same column mappings used during export.
3. Press **Load Real Data**. A success banner will confirm the number of patients and records loaded, and the patient dropdown becomes active.
4. Explore patients, generate reports, or switch to other tabs. Online training remains paused until you explicitly press **Start Online Training** in the **📊 Online Learning Monitor** tab.

## 3. Optional: reuse the artifacts elsewhere

* The CSV is compatible with most pandas- or PyTorch-based offline RL pipelines.
* The `.npz` bundle loads directly with `numpy.load`, exposing arrays named `states`, `actions`, `rewards`, `patient_ids`, `timesteps`, and `terminals`.
* The YAML schema can be referenced by the adapters in `RL0910/schema.py` to map new cohorts that follow the same column layout.

## Troubleshooting

* If you see a message about missing CPTAC assets, run `python -m cptac download` once with an internet connection.
* Errors mentioning `_ARRAY_API` typically indicate an older `pyarrow` build. The UI now disables the Arrow backend automatically, but upgrading `pyarrow` (or installing `pyarrow>=14`) will also resolve it.
* Loading failures in the UI will keep the patient dropdown disabled; re-run `OVdata.py` or double-check the schema path if that happens.

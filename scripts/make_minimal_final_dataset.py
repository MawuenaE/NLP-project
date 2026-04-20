from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_1993_tour1_binary.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_1993_tour1_binary_minimal.csv"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["id", "ocr_text", "orientation_label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")

    df_min = df[["id", "ocr_text", "orientation_label"]].copy()

    df_min = df_min.rename(columns={
        "ocr_text": "text",
        "orientation_label": "label"
    })

    # nettoyage minimal
    df_min["text"] = df_min["text"].astype(str).str.strip()
    df_min["label"] = df_min["label"].astype(str).str.strip().str.lower()

    # on enlève les textes vides éventuels
    df_min = df_min[df_min["text"].str.len() > 0].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_min.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Dataset minimal sauvegardé dans : {OUTPUT_PATH}")
    print(f"Nombre de lignes : {len(df_min)}")
    print("\nRépartition des labels :")
    print(df_min["label"].value_counts(dropna=False))
    print("\nAperçu :")
    print(df_min.head())


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "interim" / "dataset_1993_tour1.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "political_mapping_template_1993.csv"


def clean_value(x):
    if pd.isna(x):
        return "VALEUR_MANQUANTE"
    x = str(x).strip()
    return x if x else "VALEUR_MANQUANTE"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_cols = ["titulaire-soutien", "titulaire-liste"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes absentes du dataset : {missing}")

    df["titulaire-soutien"] = df["titulaire-soutien"].apply(clean_value)
    df["titulaire-liste"] = df["titulaire-liste"].apply(clean_value)

    mapping = (
        df.groupby(["titulaire-soutien", "titulaire-liste"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    mapping["orientation_label"] = ""
    mapping["keep_for_binary"] = ""
    mapping["notes"] = ""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Template sauvegardé : {OUTPUT_PATH}")
    print(f"Nombre de combinaisons uniques : {len(mapping)}")
    print("\nAperçu :")
    print(mapping.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "interim" / "dataset_1993_tour1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORT_COUNTS_PATH = OUTPUT_DIR / "titulaire_soutien_counts.csv"
LISTE_COUNTS_PATH = OUTPUT_DIR / "titulaire_liste_counts.csv"


def clean_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    )


def main():
    df = pd.read_csv(DATASET_PATH)

    if "titulaire-soutien" not in df.columns or "titulaire-liste" not in df.columns:
        raise KeyError("Les colonnes 'titulaire-soutien' et/ou 'titulaire-liste' sont absentes.")

    df["titulaire-soutien"] = clean_series(df["titulaire-soutien"])
    df["titulaire-liste"] = clean_series(df["titulaire-liste"])

    soutien_counts = (
        df["titulaire-soutien"]
        .fillna("VALEUR_MANQUANTE")
        .value_counts(dropna=False)
        .rename_axis("titulaire-soutien")
        .reset_index(name="count")
    )

    liste_counts = (
        df["titulaire-liste"]
        .fillna("VALEUR_MANQUANTE")
        .value_counts(dropna=False)
        .rename_axis("titulaire-liste")
        .reset_index(name="count")
    )

    soutien_counts.to_csv(SUPPORT_COUNTS_PATH, index=False, encoding="utf-8")
    liste_counts.to_csv(LISTE_COUNTS_PATH, index=False, encoding="utf-8")

    print("\nTop 30 - titulaire-soutien")
    print(soutien_counts.head(30).to_string(index=False))

    print("\nTop 30 - titulaire-liste")
    print(liste_counts.head(30).to_string(index=False))

    print(f"\nFichier sauvegardé : {SUPPORT_COUNTS_PATH}")
    print(f"Fichier sauvegardé : {LISTE_COUNTS_PATH}")


if __name__ == "__main__":
    main()
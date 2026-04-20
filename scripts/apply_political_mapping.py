from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "interim" / "dataset_1993_tour1.csv"
MAPPING_PATH = PROJECT_ROOT / "data" / "interim" / "political_mapping_final_1993.csv"

OUTPUT_ALL = PROJECT_ROOT / "data" / "interim" / "dataset_1993_tour1_labeled.csv"
OUTPUT_BINARY = PROJECT_ROOT / "data" / "interim" / "dataset_1993_tour1_binary.csv"
OUTPUT_EXCLUDED = PROJECT_ROOT / "data" / "interim" / "dataset_1993_tour1_excluded_from_binary.csv"


def clean_value(x):
    if pd.isna(x):
        return "VALEUR_MANQUANTE"
    x = str(x).strip()
    return x if x else "VALEUR_MANQUANTE"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping introuvable : {MAPPING_PATH}")

    df = pd.read_csv(DATASET_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    required_dataset_cols = ["titulaire-soutien", "titulaire-liste"]
    required_mapping_cols = ["titulaire-soutien", "titulaire-liste", "orientation_label", "keep_for_binary"]

    missing_dataset = [c for c in required_dataset_cols if c not in df.columns]
    missing_mapping = [c for c in required_mapping_cols if c not in mapping.columns]

    if missing_dataset:
        raise KeyError(f"Colonnes absentes du dataset : {missing_dataset}")
    if missing_mapping:
        raise KeyError(f"Colonnes absentes du mapping : {missing_mapping}")

    df["titulaire-soutien"] = df["titulaire-soutien"].apply(clean_value)
    df["titulaire-liste"] = df["titulaire-liste"].apply(clean_value)

    mapping["titulaire-soutien"] = mapping["titulaire-soutien"].apply(clean_value)
    mapping["titulaire-liste"] = mapping["titulaire-liste"].apply(clean_value)

    mapping["orientation_label"] = mapping["orientation_label"].astype(str).str.strip().str.lower()
    mapping["keep_for_binary"] = mapping["keep_for_binary"].astype(str).str.strip().str.lower()

    duplicate_keys = mapping.duplicated(subset=["titulaire-soutien", "titulaire-liste"]).sum()
    if duplicate_keys > 0:
        raise ValueError(
            f"Le mapping contient {duplicate_keys} doublons sur (titulaire-soutien, titulaire-liste)."
        )

    df_labeled = df.merge(
        mapping[["titulaire-soutien", "titulaire-liste", "orientation_label", "keep_for_binary", "notes"]],
        on=["titulaire-soutien", "titulaire-liste"],
        how="left"
    )

    OUTPUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(OUTPUT_ALL, index=False, encoding="utf-8")

    df_binary = df_labeled[
        (df_labeled["keep_for_binary"] == "yes") &
        (df_labeled["orientation_label"].isin(["left", "right"]))
    ].copy()

    df_binary.to_csv(OUTPUT_BINARY, index=False, encoding="utf-8")

    df_excluded = df_labeled[
        ~((df_labeled["keep_for_binary"] == "yes") &
          (df_labeled["orientation_label"].isin(["left", "right"])))
    ].copy()

    df_excluded.to_csv(OUTPUT_EXCLUDED, index=False, encoding="utf-8")

    mapped_count = df_labeled["orientation_label"].notna().sum()
    unlabeled = df_labeled["orientation_label"].isna().sum()

    print(f"Dataset labelisé complet : {OUTPUT_ALL}")
    print(f"Dataset binaire final : {OUTPUT_BINARY}")
    print(f"Dataset exclu du binaire : {OUTPUT_EXCLUDED}")

    print(f"\nNombre de lignes dataset complet : {len(df_labeled)}")
    print(f"Nombre de lignes dataset binaire : {len(df_binary)}")
    print(f"Lignes avec mapping appliqué : {mapped_count}")
    print(f"Lignes sans mapping appliqué : {unlabeled}")

    print("\nRépartition binaire :")
    print(df_binary["orientation_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
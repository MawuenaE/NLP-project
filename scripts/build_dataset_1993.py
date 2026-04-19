from pathlib import Path
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = PROJECT_ROOT / "data" / "interim" / "metadata_1993_tour1_filtered.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "pf_1993_tour1_manifest.csv"
OCR_DIR = PROJECT_ROOT / "data" / "raw" / "pf_1993_tour1_matched"

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DATASET = OUTPUT_DIR / "dataset_1993_tour1.csv"

# =========================================================
# HELPERS
# =========================================================
def read_text_file(path: Path) -> str:
    """Lit un fichier texte avec plusieurs encodages de secours."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    """Nettoyage léger sans dénaturer le contenu."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def build_text_dataframe_from_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in manifest_df.iterrows():
        doc_id = str(row["id"]).strip()
        file_path = Path(row["file_path"])

        if not file_path.exists():
            rows.append({
                "id": doc_id,
                "ocr_text": None,
                "ocr_char_count": None,
                "ocr_word_count": None,
                "ocr_missing_file": True
            })
            continue

        text = read_text_file(file_path)
        text = normalize_text(text)

        rows.append({
            "id": doc_id,
            "ocr_text": text,
            "ocr_char_count": len(text),
            "ocr_word_count": len(text.split()),
            "ocr_missing_file": False
        })

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {METADATA_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest introuvable : {MANIFEST_PATH}")

    if not OCR_DIR.exists():
        raise FileNotFoundError(f"Dossier OCR introuvable : {OCR_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger métadonnées
    df_meta = pd.read_csv(METADATA_PATH)
    df_meta["id"] = df_meta["id"].astype(str).str.strip()

    print(f"Nombre de lignes métadonnées : {len(df_meta)}")

    # 2. Charger manifest
    df_manifest = pd.read_csv(MANIFEST_PATH)
    df_manifest["id"] = df_manifest["id"].astype(str).str.strip()

    print(f"Nombre de lignes manifest : {len(df_manifest)}")

    # 3. Lire les OCR
    df_text = build_text_dataframe_from_manifest(df_manifest)
    df_text["id"] = df_text["id"].astype(str).str.strip()

    print(f"Nombre de lignes OCR lues : {len(df_text)}")

    # 4. Fusion metadata + OCR
    df_final = df_meta.merge(df_text, on="id", how="left", validate="one_to_one")

    print(f"Nombre de lignes après fusion : {len(df_final)}")

    # 5. Contrôles qualité
    missing_text = df_final["ocr_text"].isna().sum()
    missing_files = df_final["ocr_missing_file"].fillna(False).sum()

    print(f"Lignes sans texte OCR : {missing_text}")
    print(f"Fichiers OCR manquants physiquement : {missing_files}")

    # 6. Sauvegarde
    df_final.to_csv(OUTPUT_DATASET, index=False, encoding="utf-8")
    print(f"Dataset final sauvegardé dans : {OUTPUT_DATASET}")

    # 7. Aperçu rapide
    cols_preview = [
        c for c in [
            "id",
            "date",
            "departement-nom",
            "titulaire-nom",
            "titulaire-prenom",
            "titulaire-sexe",
            "titulaire-profession",
            "titulaire-soutien",
            "titulaire-liste",
            "ocr_char_count",
            "ocr_word_count"
        ] if c in df_final.columns
    ]

    print("\nAperçu :")
    print(df_final[cols_preview].head())


if __name__ == "__main__":
    main()
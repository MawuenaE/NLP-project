from pathlib import Path
import pandas as pd
import zipfile
import shutil
import subprocess
import sys

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "archelect_search_1993.csv"

ZIP_URL = "https://gitlab.teklia.com/ckermorvant/arkindex_archelec/-/raw/master/text_files/1993/legislatives.zip"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_DIR = RAW_DIR / "zips"
EXTRACT_DIR = RAW_DIR / "extracted_1993"
PF_DIR = RAW_DIR / "pf_1993_all"
FINAL_DIR = RAW_DIR / "pf_1993_tour1_matched"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
METADATA_OUT = INTERIM_DIR / "metadata_1993_tour1_filtered.csv"
MANIFEST_OUT = INTERIM_DIR / "pf_1993_tour1_manifest.csv"

ZIP_PATH = ZIP_DIR / "legislatives_1993.zip"

# =========================================================
# HELPERS
# =========================================================
def ensure_dirs():
    for d in [ZIP_DIR, EXTRACT_DIR, PF_DIR, FINAL_DIR, INTERIM_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def check_csv_exists():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CSV_PATH}\n"
            "Placez votre CSV à la racine du projet sous le nom : archelect_search_1993.csv"
        )


def download_with_curl(url: str, dest: Path):
    """
    Télécharge un fichier avec curl.
    On passe par curl car c'est souvent plus robuste sur les serveurs distants.
    """
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry", "3",
        "--connect-timeout", "20",
        "--max-time", "300",
        "-o", str(dest),
        url
    ]
    print(f"Téléchargement du zip vers : {dest}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Échec du téléchargement du zip avec curl.")

    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError("Le zip téléchargé est vide.")


def unzip_archive(zip_path: Path, extract_to: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive introuvable : {zip_path}")

    print(f"Dézippage de : {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def copy_pf_files(source_dir: Path, target_dir: Path):
    pf_files = list(source_dir.rglob("*_PF_*.txt"))
    print(f"Nombre de fichiers PF trouvés dans l'archive : {len(pf_files)}")

    copied = 0
    for f in pf_files:
        dest = target_dir / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            copied += 1

    print(f"Fichiers PF copiés : {copied}")


def load_and_filter_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalisation légère
    df["contexte-election"] = df["contexte-election"].astype(str).str.strip().str.lower()
    df["contexte-tour"] = pd.to_numeric(df["contexte-tour"], errors="coerce")
    df["date"] = df["date"].astype(str)

    # Filtre ciblé
    df_filtered = df[
        (df["contexte-election"] == "législatives") &
        (df["contexte-tour"] == 1) &
        (df["date"].str.contains("1993", na=False))
    ].copy()

    if "id" not in df_filtered.columns:
        raise KeyError("La colonne 'id' est absente du CSV.")

    df_filtered["id"] = df_filtered["id"].astype(str).str.strip()

    print(f"Nombre de lignes metadata retenues (1993 + législatives + tour 1) : {len(df_filtered)}")
    return df_filtered


def match_pf_with_metadata(df_meta: pd.DataFrame, pf_dir: Path, final_dir: Path) -> pd.DataFrame:
    wanted_ids = set(df_meta["id"].dropna().astype(str).str.strip())

    pf_files = list(pf_dir.glob("*.txt"))
    rows = []
    matched = 0

    for f in pf_files:
        file_id = f.stem.strip()
        if file_id in wanted_ids:
            dest = final_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

            rows.append({
                "id": file_id,
                "file_name": f.name,
                "file_path": str(dest.resolve())
            })
            matched += 1

    print(f"Nombre de fichiers PF correspondant aux métadonnées : {matched}")
    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    ensure_dirs()
    check_csv_exists()

    if not ZIP_PATH.exists():
        download_with_curl(ZIP_URL, ZIP_PATH)
    else:
        print(f"Zip déjà présent : {ZIP_PATH}")

    unzip_archive(ZIP_PATH, EXTRACT_DIR)
    copy_pf_files(EXTRACT_DIR, PF_DIR)

    df_meta = load_and_filter_metadata(CSV_PATH)
    df_meta.to_csv(METADATA_OUT, index=False, encoding="utf-8")
    print(f"Métadonnées filtrées sauvegardées dans : {METADATA_OUT}")

    manifest = match_pf_with_metadata(df_meta, PF_DIR, FINAL_DIR)
    manifest.to_csv(MANIFEST_OUT, index=False, encoding="utf-8")
    print(f"Manifest sauvegardé dans : {MANIFEST_OUT}")

    print("\nTerminé.")
    print(f"OCR finaux : {FINAL_DIR}")
    print(f"Nombre final de fichiers OCR retenus : {len(manifest)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERREUR : {e}")
        sys.exit(1)
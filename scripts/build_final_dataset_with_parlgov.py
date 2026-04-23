from pathlib import Path
import pandas as pd
import re

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_1993_tour1_binary_minimal.csv"
PARLGOV_PATH = PROJECT_ROOT / "data" / "external" / "parlgov" / "party.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_1993_final_clean.csv"


# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# MAP LEFT/RIGHT SCORE
# =========================
def map_lr(score):
    if pd.isna(score):
        return "unknown"
    if score <= 4:
        return "left"
    elif score >= 6:
        return "right"
    else:
        return "center"


# =========================
# MAIN
# =========================
def main():

    # =========================
    # 1. LOAD DATA
    # =========================
    df = pd.read_csv(DATASET_PATH)
    party_df = pd.read_csv(PARLGOV_PATH)

    print("Dataset OCR:", df.shape)
    print("ParlGov:", party_df.shape)

    print("\nColonnes ParlGov :", party_df.columns.tolist())

    # =========================
    # 2. FILTER FRANCE
    # =========================
    if "country_id" in party_df.columns:
        party_fr = party_df[party_df["country_id"] == 33].copy()
    else:
        print("⚠️ Pas de country_id → utilisation de tout le dataset")
        party_fr = party_df.copy()

    # =========================
    # 3. BUILD ORIENTATION FROM CMP
    # =========================
    if "cmp" not in party_fr.columns:
        raise ValueError("Colonne 'cmp' absente dans ParlGov")

    party_fr["orientation_ext"] = party_fr["cmp"].apply(map_lr)

    # =========================
    # 4. MANUAL MAPPING (OCR -> ParlGov)
    # =========================
    mapping_manual = pd.DataFrame([
        ["Parti socialiste", "Socialist Party"],
        ["Parti communiste français", "Communist Party"],
        ["Rassemblement pour la République", "RPR"],
        ["Union pour la démocratie française", "UDF"],
        ["Front national", "National Front"],
        ["Centre national des indépendants", "CNI"],
    ], columns=["ocr_party", "parlgov_party"])

    # =========================
    # 5. MERGE MAPPING
    # =========================
    mapping_full = mapping_manual.merge(
        party_fr[["name_english", "orientation_ext"]],
        left_on="parlgov_party",
        right_on="name_english",
        how="left"
    )

    print("\nMapping externe :")
    print(mapping_full)

    # =========================
    # 6. APPLY MAPPING
    # =========================
    df["orientation_final"] = df["label"]
    df["label_source"] = "manual"

    for _, row in mapping_full.iterrows():
        ocr_party = row["ocr_party"]
        ext_label = row["orientation_ext"]

        if pd.isna(ext_label):
            continue

        mask = df["text"].str.contains(ocr_party, case=False, na=False)

        df.loc[mask, "orientation_final"] = ext_label
        df.loc[mask, "label_source"] = "parlgov"

    print("\nRépartition après correction :")
    print(df["orientation_final"].value_counts())

    # =========================
    # 7. CLEAN TEXT
    # =========================
    df["text_clean"] = df["text"].apply(clean_text)

    # =========================
    # 8. REMOVE CONFLICTS
    # =========================
    conflicts = df.groupby("text_clean")["orientation_final"].nunique()
    conflicts = conflicts[conflicts > 1].index

    print(f"\nTextes conflictuels supprimés : {len(conflicts)}")

    df = df[~df["text_clean"].isin(conflicts)].copy()

    # =========================
    # 9. REMOVE DUPLICATES
    # =========================
    before = len(df)
    df = df.drop_duplicates(subset=["text_clean", "orientation_final"])
    after = len(df)

    print(f"Doublons supprimés : {before - after}")

    # =========================
    # 10. KEEP ONLY BINARY
    # =========================
    df = df[df["orientation_final"].isin(["left", "right"])].copy()

    # =========================
    # 11. FINAL DATASET
    # =========================
    df_final = df[["id", "text_clean", "orientation_final"]].rename(
        columns={
            "text_clean": "text",
            "orientation_final": "label"
        }
    )

    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n=== DATASET FINAL ===")
    print(df_final.shape)
    print(df_final["label"].value_counts())

    print("\nSaved at:", OUTPUT_PATH)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "political_mapping_prefilled_1993.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "political_mapping_final_1993.csv"


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def set_values(df, mask, label, keep, note):
    df.loc[mask, "orientation_label"] = label
    df.loc[mask, "keep_for_binary"] = keep
    df.loc[mask, "notes"] = note
    return df


def main():
    df = pd.read_csv(INPUT_PATH)

    df["soutien_norm"] = df["titulaire-soutien"].apply(norm)
    df["liste_norm"] = df["titulaire-liste"].apply(norm)

    # =====================================================
    # 1. Cas "ni gauche ni droite" => OTHER
    # =====================================================
    mask = df["liste_norm"].str.contains("ni de gauche ni de droite", na=False)
    df = set_values(df, mask, "other", "no", "corrigé automatiquement : ni gauche ni droite")

    mask = df["liste_norm"].str.contains("ni à droite ni à gauche", na=False)
    df = set_values(df, mask, "other", "no", "corrigé automatiquement : ni droite ni gauche")

    # =====================================================
    # 2. Cas écologistes mixtes => OTHER
    # =====================================================
    eco_mask = (
        df["soutien_norm"].str.contains("verts|écolog|ecolog|génération écologie|generation ecologie", na=False)
        | df["liste_norm"].str.contains("écolog|ecolog|entente des écologistes|union pour l'écologie|union pour l'ecologie", na=False)
    )

    radical_left_mask = df["soutien_norm"].str.contains(
        "alternative rouge et verte|solidarité écologie gauche alternative|solidarite ecologie gauche alternative",
        na=False
    )

    mask = eco_mask & radical_left_mask
    df = set_values(df, mask, "other", "no", "corrigé automatiquement : coalition écologiste mixte")

    # =====================================================
    # 3. Cas centre-droit / droite modérée => RIGHT
    # =====================================================
    mask = df["soutien_norm"].eq("centre des démocrates sociaux")
    df = set_values(df, mask, "right", "yes", "corrigé automatiquement : CDS classé centre-droit")

    mask = df["soutien_norm"].eq("démocratie chrétienne française") | df["soutien_norm"].eq("democratie chretienne francaise")
    df = set_values(df, mask, "right", "yes", "corrigé automatiquement : démocratie chrétienne classée à droite")

    mask = df["soutien_norm"].eq("parti républicain") | df["soutien_norm"].eq("parti republicain")
    df = set_values(df, mask, "right", "yes", "corrigé automatiquement : parti républicain classé à droite")

    mask = df["soutien_norm"].str.contains("centre droit", na=False)
    df = set_values(df, mask, "right", "yes", "corrigé automatiquement : centre droit agrégé à droite")

    # =====================================================
    # 4. Cas clairement gauche malgré soutien absent
    # =====================================================
    mask = df["liste_norm"].str.contains("a gauche", na=False)
    df = set_values(df, mask, "left", "yes", "corrigé automatiquement : liste explicitement à gauche")

    mask = df["liste_norm"].str.contains("changer à gauche", na=False)
    df = set_values(df, mask, "left", "yes", "corrigé automatiquement : liste explicitement à gauche")

    mask = df["liste_norm"].str.contains("oxygéner la gauche|oxygener la gauche", na=False)
    df = set_values(df, mask, "left", "yes", "corrigé automatiquement : liste explicitement à gauche")

    # =====================================================
    # 5. Cas opposition explicite + soutiens de droite => RIGHT
    # =====================================================
    mask = (
        df["liste_norm"].str.contains("opposition", na=False)
        & df["soutien_norm"].str.contains("udf|rassemblement pour la république|rpr|gaulliste|divers droite", na=False)
    )
    df = set_values(df, mask, "right", "yes", "corrigé automatiquement : opposition de droite")

    # =====================================================
    # 6. Cas trop ambigus => OTHER
    # =====================================================
    ambiguous_mask = df["soutien_norm"].isin([
        "indépendant", "independant", "sans étiquette", "sans etiquette",
        "non mentionné", "non mentionne", "non inscrit", "non inscrits",
        "apolitique", "aucun parti politique", "aucune formation politique"
    ])
    df = set_values(df, ambiguous_mask & ~df["liste_norm"].str.contains("gauche|droite|opposition|union pour la france", na=False),
                    "other", "no", "corrigé automatiquement : ambigu")

    # Nettoyage colonnes temporaires
    df = df.drop(columns=["soutien_norm", "liste_norm"])

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Mapping final sauvegardé : {OUTPUT_PATH}")
    print("\nRépartition orientation_label :")
    print(df["orientation_label"].value_counts(dropna=False))
    print("\nRépartition keep_for_binary :")
    print(df["keep_for_binary"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
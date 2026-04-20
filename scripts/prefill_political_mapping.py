from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "political_mapping_template_1993.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "political_mapping_prefilled_1993.csv"


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def contains_any(text, keywords):
    return any(kw in text for kw in keywords)


def assign_label(soutien, liste):
    s = norm(soutien)
    l = norm(liste)
    text = f"{s} || {l}"

    left_keywords = [
        "parti socialiste",
        "socialiste",
        "parti communiste français",
        "communiste",
        "lutte ouvrière",
        "ligue communiste révolutionnaire",
        "parti des travailleurs",
        "mouvement des radicaux de gauche",
        "radicaux de gauche",
        "mouvement des citoyens",
        "gauche",
        "changer à gauche",
        "a gauche vraiment",
        "forces de gauche",
        "progrès et gauche",
        "solidarité écologie gauche alternative",
        "alternative rouge et verte",
        "mouvement de la gauche progressiste",
    ]

    right_keywords = [
        "front national",
        "rassemblement pour la république",
        "union pour la démocratie française",
        "union pour la france",
        "union de l'opposition",
        "opposition",
        "divers droite",
        "droite",
        "gaulliste",
        "centre national des indépendants",
        "centre national des indépendants et paysans",
        "rassemblement des forces nationales",
        "alliance populaire",
    ]

    other_keywords = [
        "écolog",
        "ecolog",
        "verts",
        "génération écologie",
        "generation ecologie",
        "entente des écologistes",
        "union pour l'écologie et la démocratie",
        "union pour l'ecologie et la democratie",
        "sans étiquette",
        "sans etiquette",
        "indépendant",
        "independant",
        "non mentionné",
        "non mentionne",
        "apolitique",
        "aucun parti",
        "centre",
        "centriste",
        "chasse pêche nature traditions",
        "chasse peche nature traditions",
    ]

    # priorité aux cas explicitement gauche
    if contains_any(text, left_keywords):
        return "left", "yes", "pré-rempli automatiquement"

    # ensuite les cas explicitement droite
    if contains_any(text, right_keywords):
        return "right", "yes", "pré-rempli automatiquement"

    # puis les cas à exclure du binaire
    if contains_any(text, other_keywords):
        return "other", "no", "pré-rempli automatiquement"

    return "other", "no", "à revoir si besoin"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Template introuvable : {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_cols = ["titulaire-soutien", "titulaire-liste"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes absentes du mapping : {missing}")

    labels = df.apply(
        lambda row: assign_label(row["titulaire-soutien"], row["titulaire-liste"]),
        axis=1
    )

    df[["orientation_label", "keep_for_binary", "notes"]] = pd.DataFrame(
        labels.tolist(),
        index=df.index
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Mapping pré-rempli sauvegardé : {OUTPUT_PATH}")
    print("\nRépartition orientation_label :")
    print(df["orientation_label"].value_counts(dropna=False))
    print("\nRépartition keep_for_binary :")
    print(df["keep_for_binary"].value_counts(dropna=False))
    print("\nAperçu :")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
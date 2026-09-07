"""Fetch raw Luganda text datasets."""

import argparse
import os
from pathlib import Path
from datasets import load_dataset


def fetch_wikipedia_luganda(output_path: Path) -> int:
    """Fetch Luganda Wikipedia articles from Hugging Face."""
    print("Fetching Luganda Wikipedia dataset (wikimedia/wikipedia 20231101.lg)...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.lg", split="train")
    article_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            text = item.get("text", "").strip()
            if text:
                f.write(text + "\n\n")
                article_count += 1
    print(f"Saved {article_count} Wikipedia articles to {output_path}")
    return article_count


def fetch_salt_luganda(output_path: Path) -> int:
    """Fetch Sunbird SALT Luganda text if authenticated."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No HF_TOKEN detected in environment. Skipping gated Sunbird/salt.")
        return 0
    try:
        print("Fetching Sunbird/salt Luganda dataset...")
        dataset = load_dataset("Sunbird/salt", split="train", token=token)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                text = item.get("luganda", "").strip()
                if text:
                    f.write(text + "\n")
                    count += 1
        print(f"Saved {count} SALT sentences to {output_path}")
        return count
    except Exception as exc:
        print(f"Could not load Sunbird/salt: {exc}")
        return 0


def generate_fallback_corpus(output_path: Path) -> int:
    """Generate a high-quality fallback Luganda corpus for offline mode."""
    print("Generating fallback Luganda text corpus for offline testing...")
    base_sentences = [
        "Oli otya? Gyendi, webale nnyo.",
        "Uganda nsi nungi eri mu buvanjuba bwa Afirika.",
        "Kampala kye kibuga ekikulu ekya Uganda.",
        "Olulimi Oluganda lulimi lw'Abaganda.",
        "Abantu bangi mu Uganda boogera Oluganda.",
        "Enjuba eva buvanjuba n'egwa bujwanjuba.",
        "Amawulire gano gakwata ku by'obulamu n'obulimi.",
        "Abalimi balima emwanyi, amatooke, amataffaali n'ebijanjaalo.",
        "Abayizi bagenda ku masomero okufuna amagezi n'obukugu.",
        "Abaana bazaanyira ku bisaawe by'essomero buli lunaku olw'omukaaga.",
        "Obulamu bwaffe bwesigamye ku mmere ennungi n'amazzi amayonjo.",
        "Gavumenti etaddewo enteekateeka z'okukulaakulanya ebyalo.",
        "Tekinologiya w'ebweru ayamba nnyo mu kukolera awamu n'okuyiga.",
        "Enkuba bweetonnya, ebirime bikula bulungi n'ebibira biba bya kiragala.",
        "Omuntu yenna asaanidde okuyiga okuwandiika n'okusoma olulimi lwe.",
        "Pearl-Chat nkola ey'amagezi ag'obukugu efunze okulondoola olulimi Oluganda.",
        "JAX nkola ya sayansi eya kompyuta ewa amanyi mangi mu kubala.",
        "Flax nnx etegeka obutundu bw'omutwe gw'amagezi ag'obukugu obulungi.",
        "Kino kitundu kikulu mu lukuŋŋaana lwa PyCon Africa.",
        "Tukola emikolo gino okusitula olulimi n'amagezi mu nsi yaffe.",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for cycle in range(500):
            for sentence in base_sentences:
                f.write(f"{sentence}\n")
                count += 1
    print(f"Saved {count} fallback sentences to {output_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Luganda text data.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to store raw data files",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    wiki_path = raw_path / "wikipedia_luganda.txt"
    salt_path = raw_path / "salt_luganda.txt"

    fetched_any = False

    try:
        wiki_count = fetch_wikipedia_luganda(wiki_path)
        if wiki_count > 0:
            fetched_any = True
    except Exception as exc:
        print(f"Wikipedia fetch encountered an error: {exc}")

    try:
        salt_count = fetch_salt_luganda(salt_path)
        if salt_count > 0:
            fetched_any = True
    except Exception as exc:
        print(f"SALT fetch encountered an error: {exc}")

    if not fetched_any:
        fallback_path = raw_path / "fallback_luganda.txt"
        generate_fallback_corpus(fallback_path)

    print("Data fetch completed successfully.")


if __name__ == "__main__":
    main()

import json
import os
from SoccerNet.Downloader import SoccerNetDownloader

# ==========================================
# PATHS
# ==========================================

SELECTION_FILE = r"E:\SprotsHighlights\datasets\SoccerNet\selected_50_training_matches.json"

LOCAL_DIRECTORY = r"E:\SprotsHighlights\datasets\SoccerNet"


# ==========================================
# LOAD SELECTED MATCHES
# ==========================================

with open(SELECTION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

additional_matches = data["additional_matches"]

print("Additional matches:", len(additional_matches))


# ==========================================
# CREATE DOWNLOADER
# ==========================================

downloader = SoccerNetDownloader(
    LocalDirectory=LOCAL_DIRECTORY
)


# ==========================================
# DOWNLOAD ONE MATCH AT A TIME
# ==========================================

for index, match_path in enumerate(additional_matches, start=1):

    # Convert full Windows path into SoccerNet game identifier
    relative_path = os.path.relpath(
        match_path,
        LOCAL_DIRECTORY
    )

    print("\n" + "=" * 70)
    print(f"Downloading match {index}/{len(additional_matches)}")
    print(relative_path)
    print("=" * 70)

    try:

        downloader.downloadGame(
            game=relative_path,
            files=[
                "1_224p.mkv",
                "2_224p.mkv"
            ],
            spl="train",
            verbose=True,
            source="HuggingFace"
        )

        print(f"Completed: {index}/{len(additional_matches)}")

    except Exception as e:

        print("\nERROR downloading:")
        print(relative_path)

        print("Error:")
        print(e)

        print("\nContinuing with next match...")


print("\n" + "=" * 70)
print("DOWNLOAD PROCESS FINISHED")
print("=" * 70)
import json
import os
from SoccerNet.Downloader import SoccerNetDownloader

ROOT = r"E:\SprotsHighlights\datasets\SoccerNet"

SELECTION_FILE = os.path.join(
    ROOT,
    "selected_training_matches.json"
)

# Load selected matches
with open(
    SELECTION_FILE,
    "r",
    encoding="utf-8"
) as f:
    matches = json.load(f)

print("Matches selected:", len(matches))
print()

# SoccerNet downloader
downloader = SoccerNetDownloader(
    LocalDirectory=ROOT
)

# Download each match individually
for number, game in enumerate(matches, start=1):

    print("=" * 60)
    print(f"Downloading match {number}/{len(matches)}")
    print(game)
    print("=" * 60)

    try:

        downloader.downloadGame(
            game=game.replace(ROOT + os.sep, "").replace("\\", "/"),
            files=[
                "1_224p.mkv",
                "2_224p.mkv"
            ],
            spl="train"
        )

        print("DOWNLOAD COMPLETED")

    except Exception as e:

        print("DOWNLOAD FAILED")
        print("Error:", e)

    print()

print("=" * 60)
print("DOWNLOAD PROCESS FINISHED")
print("=" * 60)
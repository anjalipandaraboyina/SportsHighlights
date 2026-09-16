import os
import json

SOCCERNET_ROOT = r"E:\SprotsHighlights\datasets\SoccerNet"
OUTPUT_FILE = r"E:\SprotsHighlights\datasets\SoccerNet\selected_50_training_matches.json"

# Find all matches containing Labels-v2.json
all_matches = []

for root, dirs, files in os.walk(SOCCERNET_ROOT):

    if "Labels-v2.json" in files:

        match_folder = root

        all_matches.append(match_folder)


# Sort for reproducibility
all_matches.sort()


print("Total matches with labels:", len(all_matches))


# Existing downloaded matches
existing_matches = []

for match_folder in all_matches:

    video1 = os.path.join(match_folder, "1_224p.mkv")
    video2 = os.path.join(match_folder, "2_224p.mkv")

    if os.path.exists(video1) and os.path.exists(video2):

        existing_matches.append(match_folder)


print("Already downloaded complete matches:", len(existing_matches))


# Select first 20 existing matches
existing_matches = existing_matches[:20]


# Select additional matches that are not already downloaded
additional_matches = []

for match_folder in all_matches:

    if match_folder in existing_matches:
        continue

    additional_matches.append(match_folder)

    if len(additional_matches) == 30:
        break


# Final 50 matches
final_matches = existing_matches + additional_matches


print("\nExisting matches:", len(existing_matches))
print("Additional matches:", len(additional_matches))
print("Total selected:", len(final_matches))


# Create information
data = {
    "total_matches": len(final_matches),
    "existing_matches": existing_matches,
    "additional_matches": additional_matches,
    "all_matches": final_matches
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        data,
        f,
        indent=4
    )


print("\nSelection saved to:")
print(OUTPUT_FILE)


print("\nAdditional matches to download:")

for i, match in enumerate(additional_matches, start=1):

    print(f"{i}. {match}")
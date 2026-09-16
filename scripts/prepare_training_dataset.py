import os
import json
import csv

ROOT = r"E:\SprotsHighlights\datasets\SoccerNet"

SELECTION_FILE = os.path.join(
    ROOT,
    "selected_training_matches.json"
)

OUTPUT_DIR = r"E:\SprotsHighlights\datasets\annotations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VALID_CSV = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")


# --------------------------------------------------
# Convert SoccerNet gameTime into seconds
# --------------------------------------------------

def game_time_to_seconds(game_time):

    parts = game_time.split(" - ")

    half = int(parts[0])
    time_part = parts[1]

    minutes, seconds = time_part.split(":")

    total_seconds = (
        int(minutes) * 60 +
        int(seconds)
    )

    return total_seconds


# --------------------------------------------------
# Load selected matches
# --------------------------------------------------

with open(
    SELECTION_FILE,
    "r",
    encoding="utf-8"
) as file:

    matches = json.load(file)


print("Selected matches:", len(matches))
print()


# --------------------------------------------------
# Prepare CSV files
# --------------------------------------------------

headers = [
    "video_path",
    "match_id",
    "half",
    "event_time",
    "start_time",
    "end_time",
    "label"
]


train_rows = []
valid_rows = []
test_rows = []


# --------------------------------------------------
# Process each match
# --------------------------------------------------

for index, match_path in enumerate(matches, start=1):

    print("=" * 70)
    print(f"Processing match {index}/{len(matches)}")
    print(match_path)

    label_file = os.path.join(
        match_path,
        "Labels-v2.json"
    )

    if not os.path.exists(label_file):

        print("Labels file not found.")
        continue


    # ----------------------------------------------
    # Decide dataset split
    # ----------------------------------------------

    if index <= 15:

        dataset_split = "train"

    elif index <= 18:

        dataset_split = "validation"

    else:

        dataset_split = "test"


    # ----------------------------------------------
    # Read labels
    # ----------------------------------------------

    with open(
        label_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)


    annotations = data["annotations"]

    match_id = os.path.basename(match_path)


    # ----------------------------------------------
    # Process annotations
    # ----------------------------------------------

    for annotation in annotations:

        label = annotation["label"]

        game_time = annotation["gameTime"]

        event_time = game_time_to_seconds(game_time)

        half = int(game_time.split(" - ")[0])


        # ------------------------------------------
        # Select correct video
        # ------------------------------------------

        if half == 1:

            video_path = os.path.join(
                match_path,
                "1_224p.mkv"
            )

        else:

            video_path = os.path.join(
                match_path,
                "2_224p.mkv"
            )


        if not os.path.exists(video_path):

            continue


        # ------------------------------------------
        # Create temporal window
        # ------------------------------------------

        start_time = max(
            0,
            event_time - 10
        )

        end_time = event_time + 10


        row = [
            video_path,
            match_id,
            half,
            event_time,
            start_time,
            end_time,
            label
        ]


        if dataset_split == "train":

            train_rows.append(row)

        elif dataset_split == "validation":

            valid_rows.append(row)

        else:

            test_rows.append(row)


# --------------------------------------------------
# Write CSV files
# --------------------------------------------------

def write_csv(filename, rows):

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        writer.writerows(rows)


write_csv(
    TRAIN_CSV,
    train_rows
)

write_csv(
    VALID_CSV,
    valid_rows
)

write_csv(
    TEST_CSV,
    test_rows
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("=" * 70)

print("DATASET PREPARATION COMPLETE")

print("=" * 70)

print()

print("Training samples   :", len(train_rows))
print("Validation samples :", len(valid_rows))
print("Test samples       :", len(test_rows))

print()

print("Files created:")

print(TRAIN_CSV)
print(VALID_CSV)
print(TEST_CSV)

print()
print("Finished.")
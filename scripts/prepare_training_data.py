import json
import os
import subprocess
import re

GAME_FOLDER = r"E:\SprotsHighlights\datasets\SoccerNet\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

LABEL_FILE = os.path.join(
    GAME_FOLDER,
    "Labels-v2.json"
)

OUTPUT_FOLDER = r"E:\SprotsHighlights\datasets\train_samples"

WINDOW_BEFORE = 10
WINDOW_AFTER = 10

SELECTED_EVENTS = [
    "Goal",
    "Corner",
    "Foul",
    "Shots on target",
    "Shots off target",
    "Penalty",
    "Red card",
    "Yellow card",
    "Substitution",
    "Offside",
    "Direct free-kick",
    "Indirect free-kick"
]


def convert_time(game_time):

    parts = game_time.split(" - ")

    half = int(parts[0])

    minutes, seconds = parts[1].split(":")

    total_seconds = (
        int(minutes) * 60 +
        int(seconds)
    )

    return half, total_seconds


with open(
    LABEL_FILE,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


annotations = data["annotations"]

print("Total annotations:", len(annotations))

created = 0


for annotation in annotations:

    label = annotation["label"]

    if label not in SELECTED_EVENTS:
        continue

    half, event_time = convert_time(
        annotation["gameTime"]
    )

    if half == 1:

        video = os.path.join(
            GAME_FOLDER,
            "1_224p.mkv"
        )

    else:

        video = os.path.join(
            GAME_FOLDER,
            "2_224p.mkv"
        )

    safe_label = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        label
    ).strip("_")

    class_folder = os.path.join(
        OUTPUT_FOLDER,
        safe_label
    )

    os.makedirs(
        class_folder,
        exist_ok=True
    )

    start_time = max(
        0,
        event_time - WINDOW_BEFORE
    )

    output_file = os.path.join(
        class_folder,
        f"half{half}_{event_time}.mp4"
    )

    if os.path.exists(output_file):
        continue

    duration = (
        WINDOW_BEFORE +
        WINDOW_AFTER
    )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        video,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        output_file
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:

        created += 1

        print(
            created,
            "|",
            label,
            "|",
            annotation["gameTime"]
        )

    else:

        print(
            "FAILED:",
            label,
            annotation["gameTime"]
        )


print()
print("==============================")
print("TRAINING DATA PREPARATION DONE")
print("==============================")
print("Samples created:", created)
print("Location:", OUTPUT_FOLDER)
import json
import os
import subprocess
import re

# -----------------------------
# PATHS
# -----------------------------

GAME_FOLDER = r"E:\SprotsHighlights\datasets\SoccerNet\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

LABEL_FILE = os.path.join(GAME_FOLDER, "Labels-v2.json")

OUTPUT_FOLDER = r"E:\SprotsHighlights\outputs\events"

# -----------------------------
# SETTINGS
# -----------------------------

BEFORE_SECONDS = 10
AFTER_SECONDS = 10

# Events we want to extract
SELECTED_EVENTS = [
    "Goal",
    "Corner",
    "Foul",
    "Shots on target",
    "Shots off target",
    "Penalty",
    "Red card",
    "Yellow card",
    "Yellow->red card",
    "Substitution",
    "Offside",
    "Direct free-kick",
    "Indirect free-kick"
]

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# LOAD LABELS
# -----------------------------

with open(LABEL_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

annotations = data["annotations"]

print("Total annotations:", len(annotations))
print()

# -----------------------------
# CONVERT GAME TIME
# -----------------------------

def time_to_seconds(game_time):

    # Example:
    # 1 - 02:13
    # 2 - 45:30

    parts = game_time.split(" - ")

    half = int(parts[0])
    time_part = parts[1]

    minutes, seconds = time_part.split(":")

    minutes = int(minutes)
    seconds = int(seconds)

    total_seconds = minutes * 60 + seconds

    return half, total_seconds


# -----------------------------
# PROCESS EVENTS
# -----------------------------

event_number = 1

for annotation in annotations:

    label = annotation["label"]

    if label not in SELECTED_EVENTS:
        continue

    game_time = annotation["gameTime"]

    half, event_seconds = time_to_seconds(game_time)

    # Select correct video
    if half == 1:
        video_file = os.path.join(GAME_FOLDER, "1_224p.mkv")
    else:
        video_file = os.path.join(GAME_FOLDER, "2_224p.mkv")

    # Start time
    start_seconds = max(0, event_seconds - BEFORE_SECONDS)

    # Output filename
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label)

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"{event_number:03d}_{safe_label}_{half}_{event_seconds}.mp4"
    )

    print("Creating:", output_file)
    print("Event:", label)
    print("Half:", half)
    print("Time:", event_seconds, "seconds")

    # FFmpeg command
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_seconds),
        "-i",
        video_file,
        "-t",
        str(BEFORE_SECONDS + AFTER_SECONDS),
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
        print("SUCCESS")
    else:
        print("FAILED")

    print()

    event_number += 1

print("================================")
print("Finished creating event clips.")
print("Output folder:")
print(OUTPUT_FOLDER)
print("================================")
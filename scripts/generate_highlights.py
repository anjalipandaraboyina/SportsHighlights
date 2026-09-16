import os
import re
import subprocess

# --------------------------------------------------
# PATHS
# --------------------------------------------------

EVENT_FOLDER = r"E:\SprotsHighlights\outputs\events"

OUTPUT_FOLDER = r"E:\SprotsHighlights\outputs\highlights"

FULL_HIGHLIGHT = os.path.join(
    OUTPUT_FOLDER,
    "full_highlights.mp4"
)

REELS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "reels"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REELS_FOLDER, exist_ok=True)


# --------------------------------------------------
# EVENT IMPORTANCE SCORES
# --------------------------------------------------

EVENT_SCORES = {
    "Goal": 100,
    "Red card": 90,
    "Yellow->red card": 85,
    "Penalty": 80,
    "Shots on target": 65,
    "Shots off target": 40,
    "Substitution": 35,
    "Direct free-kick": 30,
    "Indirect free-kick": 25,
    "Corner": 25,
    "Offside": 20,
    "Foul": 15,
}


# --------------------------------------------------
# READ EVENT FILES
# --------------------------------------------------

events = []

for filename in os.listdir(EVENT_FOLDER):

    if not filename.lower().endswith(".mp4"):
        continue

    # Example:
    # 057_Goal_2_2121.mp4

    match = re.match(
        r"(\d+)_(.+?)_(\d+)_(\d+)\.mp4$",
        filename
    )

    if not match:
        continue

    number = int(match.group(1))
    event_name = match.group(2)
    half = int(match.group(3))
    time_seconds = int(match.group(4))

    # Convert underscores back to spaces
    event_name = event_name.replace("_", " ")

    # Get score
    score = EVENT_SCORES.get(event_name, 10)

    events.append({
        "filename": filename,
        "number": number,
        "event": event_name,
        "half": half,
        "time": time_seconds,
        "score": score
    })


print("Total event clips found:", len(events))


# --------------------------------------------------
# SORT BY IMPORTANCE
# --------------------------------------------------

events.sort(
    key=lambda x: x["score"],
    reverse=True
)


print()
print("Events ranked by importance:")
print("--------------------------------")

for event in events:
    print(
        event["event"],
        "| Score:",
        event["score"],
        "| Half:",
        event["half"],
        "| Time:",
        event["time"]
    )


# --------------------------------------------------
# REMOVE OVERLAPPING / REPEATED EVENTS
# --------------------------------------------------

selected = []

for event in events:

    duplicate = False

    for chosen in selected:

        # Different halves cannot overlap
        if event["half"] != chosen["half"]:
            continue

        # Each clip is approximately 20 seconds
        time_difference = abs(
            event["time"] - chosen["time"]
        )

        if time_difference < 20:

            duplicate = True
            break

    if not duplicate:
        selected.append(event)


# --------------------------------------------------
# LIMIT NUMBER OF HIGHLIGHTS
# --------------------------------------------------

MAX_HIGHLIGHTS = 20

selected = selected[:MAX_HIGHLIGHTS]


# --------------------------------------------------
# CHRONOLOGICAL ORDER
# --------------------------------------------------

selected.sort(
    key=lambda x: (
        x["half"],
        x["time"]
    )
)


print()
print("--------------------------------")
print("SELECTED HIGHLIGHTS")
print("--------------------------------")

for event in selected:

    print(
        event["event"],
        "| Half:",
        event["half"],
        "| Time:",
        event["time"],
        "| Score:",
        event["score"]
    )


# --------------------------------------------------
# CREATE CONCAT FILE
# --------------------------------------------------

concat_file = os.path.join(
    OUTPUT_FOLDER,
    "highlight_list.txt"
)

with open(concat_file, "w", encoding="utf-8") as file:

    for event in selected:

        path = os.path.join(
            EVENT_FOLDER,
            event["filename"]
        )

        # FFmpeg concat format
        file.write(
            "file '" +
            path.replace("\\", "/") +
            "'\n"
        )


# --------------------------------------------------
# CREATE FULL HIGHLIGHTS VIDEO
# --------------------------------------------------

print()
print("Creating full highlights video...")

command = [
    "ffmpeg",
    "-y",
    "-f",
    "concat",
    "-safe",
    "0",
    "-i",
    concat_file,
    "-c:v",
    "libx264",
    "-preset",
    "fast",
    "-c:a",
    "aac",
    FULL_HIGHLIGHT
]

result = subprocess.run(command)

if result.returncode == 0:

    print()
    print("FULL HIGHLIGHTS CREATED:")
    print(FULL_HIGHLIGHT)

else:

    print("Failed to create full highlights video.")


# --------------------------------------------------
# CREATE INDIVIDUAL REELS
# --------------------------------------------------

print()
print("Creating individual reels...")

for index, event in enumerate(selected, start=1):

    source = os.path.join(
        EVENT_FOLDER,
        event["filename"]
    )

    reel_name = (
        f"reel_{index:02d}_"
        f"{event['event'].replace(' ', '_')}.mp4"
    )

    output = os.path.join(
        REELS_FOLDER,
        reel_name
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        source,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        output
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:

        print(
            "Created:",
            reel_name
        )

    else:

        print(
            "Failed:",
            reel_name
        )


# --------------------------------------------------
# FINISHED
# --------------------------------------------------

print()
print("======================================")
print("HIGHLIGHT GENERATION COMPLETED")
print("======================================")

print()
print("Full highlights:")
print(FULL_HIGHLIGHT)

print()
print("Individual reels:")
print(REELS_FOLDER)
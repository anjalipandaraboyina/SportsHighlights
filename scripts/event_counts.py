import json
import glob

base_path = r"E:\SprotsHighlights\datasets\SoccerNet"

files = glob.glob(
    base_path + r"\**\Labels-v2.json",
    recursive=True
)

event_counts = {}

for file in files:

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data.get("annotations", []):

        label = item.get("label")

        if label:
            if label not in event_counts:
                event_counts[label] = 0

            event_counts[label] += 1

print("\nEvent counts:")
print("-" * 40)

for label in sorted(event_counts):
    print(f"{label}: {event_counts[label]}")

print("-" * 40)
print("Total events:", sum(event_counts.values()))
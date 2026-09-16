import os
import json

ROOT = r"E:\SprotsHighlights\datasets\SoccerNet"

matches = []

for root, dirs, files in os.walk(ROOT):

    if "Labels-v2.json" in files:

        matches.append(root)


matches = sorted(matches)

print("Total matches with labels:", len(matches))
print()

# Select first 20 matches
selected = matches[:20]

print("Selected matches:")
print("==============================")

for i, match in enumerate(selected, start=1):

    if i <= 15:
        split = "TRAIN"
    elif i <= 18:
        split = "VALIDATION"
    else:
        split = "TEST"

    print(f"{i:02d} [{split}] {match}")


# Save selection
output_file = os.path.join(
    ROOT,
    "selected_training_matches.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        selected,
        file,
        indent=4
    )


print()
print("Saved selection to:")
print(output_file)
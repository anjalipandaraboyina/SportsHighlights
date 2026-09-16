import json
import glob

base_path = r"E:\SprotsHighlights\datasets\SoccerNet"

files = glob.glob(
    base_path + r"\**\Labels-v2.json",
    recursive=True
)

labels = set()

for file in files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("annotations", []):
            label = item.get("label")

            if label:
                labels.add(label)

    except Exception as e:
        print("Error:", file)
        print(e)

print("\nTotal label files:", len(files))
print("Total unique event labels:", len(labels))

print("\nAvailable event labels:")
for label in sorted(labels):
    print(label)
import json
import glob

base_path = r"E:\SprotsHighlights\datasets\SoccerNet"

files = glob.glob(
    base_path + r"\**\Labels-v2.json",
    recursive=True
)

file = files[0]

print("Inspecting:")
print(file)

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\nTop-level keys:")
print(data.keys())

print("\nFirst 5 annotations:\n")

for item in data.get("annotations", [])[:5]:
    print(item)
    print()
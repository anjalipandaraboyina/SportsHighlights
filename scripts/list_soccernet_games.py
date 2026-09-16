import os

ROOT = r"E:\SprotsHighlights\datasets\SoccerNet"

for root, dirs, files in os.walk(ROOT):

    if "Labels-v2.json" in files:

        print(root)
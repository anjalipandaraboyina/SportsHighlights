import csv
from collections import Counter

FILES = [
    (
        r"E:\SprotsHighlights\datasets\training_samples\train_samples.csv",
        "TRAIN"
    ),
    (
        r"E:\SprotsHighlights\datasets\training_samples\validation_samples.csv",
        "VALIDATION"
    ),
    (
        r"E:\SprotsHighlights\datasets\training_samples\test_samples.csv",
        "TEST"
    )
]


for filename, name in FILES:

    counter = Counter()

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            counter[row["label"]] += 1


    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    for label, count in sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{label:<25} {count}"
        )
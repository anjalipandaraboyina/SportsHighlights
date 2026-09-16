import os
import csv
import random


ROOT = r"E:\SprotsHighlights\datasets"

ANNOTATIONS_DIR = os.path.join(
    ROOT,
    "annotations"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "training_samples"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TRAIN_FILE = os.path.join(
    ANNOTATIONS_DIR,
    "train.csv"
)

VALIDATION_FILE = os.path.join(
    ANNOTATIONS_DIR,
    "validation.csv"
)

TEST_FILE = os.path.join(
    ANNOTATIONS_DIR,
    "test.csv"
)


TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_samples.csv"
)

VALIDATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "validation_samples.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "test_samples.csv"
)


# --------------------------------------------------
# Settings
# --------------------------------------------------

WINDOW_LENGTH = 20

NEGATIVE_SAMPLES_PER_POSITIVE = 1

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# --------------------------------------------------
# Read CSV
# --------------------------------------------------

def read_csv(filename):

    rows = []

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append(row)

    return rows


# --------------------------------------------------
# Get video duration
# --------------------------------------------------

def get_video_duration(video_path):

    import cv2

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():

        return 0

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    capture.release()

    if fps <= 0:

        return 0

    return frame_count / fps


# --------------------------------------------------
# Create negative samples
# --------------------------------------------------

def create_negative_samples(rows):

    negatives = []

    # Group events by video
    videos = {}

    for row in rows:

        video = row["video_path"]

        event_time = float(
            row["event_time"]
        )

        if video not in videos:

            videos[video] = []

        videos[video].append(
            event_time
        )


    # Generate background samples
    for video_path, event_times in videos.items():

        duration = get_video_duration(
            video_path
        )

        if duration <= WINDOW_LENGTH:

            continue


        positive_count = len(
            event_times
        )

        required_negatives = (
            positive_count *
            NEGATIVE_SAMPLES_PER_POSITIVE
        )

        attempts = 0

        created = 0

        while (
            created < required_negatives
            and attempts < required_negatives * 20
        ):

            attempts += 1

            start_time = random.uniform(
                0,
                duration - WINDOW_LENGTH
            )

            end_time = (
                start_time +
                WINDOW_LENGTH
            )

            center_time = (
                start_time +
                WINDOW_LENGTH / 2
            )


            # Check distance from every event
            too_close = False

            for event_time in event_times:

                distance = abs(
                    center_time -
                    event_time
                )

                if distance < 15:

                    too_close = True

                    break


            if too_close:

                continue


            negatives.append({
                "video_path": video_path,
                "match_id": "",
                "half": "",
                "event_time": "",
                "start_time": start_time,
                "end_time": end_time,
                "label": "Background"
            })

            created += 1


    return negatives


# --------------------------------------------------
# Create final dataset
# --------------------------------------------------

def create_dataset(input_file, output_file):

    print()
    print("=" * 70)

    print("Processing:")
    print(input_file)

    positive_rows = read_csv(
        input_file
    )

    print(
        "Positive samples:",
        len(positive_rows)
    )


    negative_rows = create_negative_samples(
        positive_rows
    )

    print(
        "Negative samples:",
        len(negative_rows)
    )


    all_rows = (
        positive_rows +
        negative_rows
    )


    random.shuffle(
        all_rows
    )


    headers = [
        "video_path",
        "match_id",
        "half",
        "event_time",
        "start_time",
        "end_time",
        "label"
    ]


    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )


    print(
        "Total samples:",
        len(all_rows)
    )

    print(
        "Saved:",
        output_file
    )


# --------------------------------------------------
# Create train dataset
# --------------------------------------------------

create_dataset(
    TRAIN_FILE,
    TRAIN_OUTPUT
)


# --------------------------------------------------
# Create validation dataset
# --------------------------------------------------

create_dataset(
    VALIDATION_FILE,
    VALIDATION_OUTPUT
)


# --------------------------------------------------
# Create test dataset
# --------------------------------------------------

create_dataset(
    TEST_FILE,
    TEST_OUTPUT
)


print()
print("=" * 70)
print("TRAINING SAMPLE CREATION COMPLETE")
print("=" * 70)

print()
print("Training:")
print(TRAIN_OUTPUT)

print()
print("Validation:")
print(VALIDATION_OUTPUT)

print()
print("Testing:")
print(TEST_OUTPUT)
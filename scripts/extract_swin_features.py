import os
import csv
import torch

from video_dataset import (
    SportsVideoDataset
)

from torchvision.models import (
    swin_t,
    Swin_T_Weights
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = r"E:\SprotsHighlights"

TRAIN_CSV = (
    ROOT +
    r"\datasets\training_samples\train_samples.csv"
)

VALIDATION_CSV = (
    ROOT +
    r"\datasets\training_samples\validation_samples.csv"
)

TEST_CSV = (
    ROOT +
    r"\datasets\training_samples\test_samples.csv"
)

FEATURE_DIR = (
    ROOT +
    r"\datasets\swin_features"
)

os.makedirs(
    FEATURE_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load pretrained Swin-Tiny
# --------------------------------------------------

print()
print("Loading Swin Transformer...")


weights = Swin_T_Weights.DEFAULT

model = swin_t(
    weights=weights
)


# Remove ImageNet classifier

model.head = torch.nn.Identity()


model = model.to(device)

model.eval()


print("Swin Transformer ready.")


# --------------------------------------------------
# Process one CSV
# --------------------------------------------------

def extract_features(
    csv_file,
    output_file,
    name
):

    print()
    print("=" * 70)

    print(
        "Processing:",
        name
    )

    print("=" * 70)


    dataset = SportsVideoDataset(
        csv_file,
        frames_per_clip=8
    )


    all_features = []

    all_labels = []


    total = len(dataset)


    for index in range(total):

        frames, label = dataset[index]


        # frames:
        # [8, 3, 224, 224]

        frames = frames.to(
            device
        )


        with torch.no_grad():

            features = model(
                frames
            )


        # Expected:
        # [8, 768]

        features = features.cpu()


        all_features.append(
            features
        )

        all_labels.append(
            label
        )


        if (
            (index + 1) % 100 == 0
            or index == total - 1
        ):

            print(
                f"Processed {index + 1}/{total}"
            )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    features_tensor = torch.stack(
        all_features
    )

    labels_tensor = torch.stack(
        all_labels
    )


    torch.save(
        {
            "features": features_tensor,
            "labels": labels_tensor
        },
        output_file
    )


    print()
    print("Saved:")
    print(output_file)

    print()
    print("Feature shape:")

    print(
        features_tensor.shape
    )

    print()
    print("Label shape:")

    print(
        labels_tensor.shape
    )


# --------------------------------------------------
# Train
# --------------------------------------------------

extract_features(
    TRAIN_CSV,
    os.path.join(
        FEATURE_DIR,
        "train_features.pt"
    ),
    "TRAIN"
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

extract_features(
    VALIDATION_CSV,
    os.path.join(
        FEATURE_DIR,
        "validation_features.pt"
    ),
    "VALIDATION"
)


# --------------------------------------------------
# Test
# --------------------------------------------------

extract_features(
    TEST_CSV,
    os.path.join(
        FEATURE_DIR,
        "test_features.pt"
    ),
    "TEST"
)


print()
print("=" * 70)
print("SWIN FEATURE EXTRACTION COMPLETE")
print("=" * 70)
import os
import torch

from video_dataset import (
    SportsVideoDataset,
    CLASSES
)


CSV_FILE = (
    r"E:\SprotsHighlights\datasets"
    r"\training_samples\train_samples.csv"
)


print("Creating dataset...")


dataset = SportsVideoDataset(
    CSV_FILE,
    frames_per_clip=8
)


print()
print("Number of samples:", len(dataset))

print()
print("Number of classes:", len(CLASSES))

print()
print("Classes:")

for index, name in enumerate(CLASSES):

    print(
        index,
        "->",
        name
    )


print()
print("Reading first sample...")


frames, label = dataset[0]


print()
print("Frames shape:")
print(frames.shape)


print()
print("Label:")
print(label.item())


print()
print("Label name:")
print(
    CLASSES[label.item()]
)


print()
print("Dataset test successful.")
import csv
import cv2
import torch

from torch.utils.data import Dataset
from torchvision import transforms


# --------------------------------------------------
# Classes for the first model
# --------------------------------------------------

CLASSES = [
    "Background",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Clearance",
    "Indirect free-kick",
    "Shots on target",
    "Corner",
    "Shots off target",
    "Substitution",
    "Kick-off",
    "Direct free-kick",
    "Offside",
    "Yellow card",
    "Goal"
]


CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASSES)
}


# --------------------------------------------------
# Image transformation
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class SportsVideoDataset(Dataset):

    def __init__(
        self,
        csv_file,
        frames_per_clip=8
    ):

        self.frames_per_clip = frames_per_clip

        self.samples = []


        # ------------------------------------------
        # Read CSV
        # ------------------------------------------

        with open(
            csv_file,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                label = row["label"]


                # ----------------------------------
                # Ignore rare classes for now
                # ----------------------------------

                if label not in CLASS_TO_INDEX:

                    continue


                self.samples.append({
                    "video_path": row["video_path"],

                    "start_time": float(
                        row["start_time"]
                    ),

                    "end_time": float(
                        row["end_time"]
                    ),

                    "label": CLASS_TO_INDEX[
                        label
                    ]
                })


        print(
            "Loaded samples:",
            len(self.samples)
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        sample = self.samples[index]


        video_path = sample[
            "video_path"
        ]

        start_time = sample[
            "start_time"
        ]

        end_time = sample[
            "end_time"
        ]


        label = sample[
            "label"
        ]


        # ------------------------------------------
        # Open video
        # ------------------------------------------

        capture = cv2.VideoCapture(
            video_path
        )


        if not capture.isOpened():

            raise RuntimeError(
                f"Could not open video:\n{video_path}"
            )


        fps = capture.get(
            cv2.CAP_PROP_FPS
        )


        if fps <= 0:

            capture.release()

            raise RuntimeError(
                f"Invalid FPS:\n{video_path}"
            )


        # ------------------------------------------
        # Sample frame positions
        # ------------------------------------------

        duration = (
            end_time -
            start_time
        )


        frames = []


        for i in range(
            self.frames_per_clip
        ):

            ratio = (
                i /
                (self.frames_per_clip - 1)
            )


            timestamp = (
                start_time +
                ratio * duration
            )


            frame_number = int(
                timestamp * fps
            )


            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )


            success, frame = (
                capture.read()
            )


            if not success:

                continue


            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            frame = transform(
                frame
            )


            frames.append(
                frame
            )


        capture.release()


        # ------------------------------------------
        # Make sure we have 8 frames
        # ------------------------------------------

        if len(frames) == 0:

            raise RuntimeError(
                "No frames could be read from:\n"
                + video_path
            )


        while len(frames) < self.frames_per_clip:

            frames.append(
                frames[-1].clone()
            )


        frames = torch.stack(
            frames
        )


        return frames, torch.tensor(
            label,
            dtype=torch.long
        )
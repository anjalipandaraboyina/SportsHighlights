import os
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# ==================================================
# Paths
# ==================================================

ROOT = r"E:\SprotsHighlights"

FEATURE_DIR = os.path.join(
    ROOT,
    "datasets",
    "swin_features"
)

MODEL_DIR = os.path.join(
    ROOT,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


TRAIN_FILE = os.path.join(
    FEATURE_DIR,
    "train_features.pt"
)

VALIDATION_FILE = os.path.join(
    FEATURE_DIR,
    "validation_features.pt"
)

TEST_FILE = os.path.join(
    FEATURE_DIR,
    "test_features.pt"
)


BEST_MODEL = os.path.join(
    MODEL_DIR,
    "swin_lstm_event_model.pth"
)


# ==================================================
# Settings
# ==================================================

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 0.001

INPUT_SIZE = 768

HIDDEN_SIZE = 256

NUM_LAYERS = 2

NUM_CLASSES = 15

DROPOUT = 0.3


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cpu"
)

print("Device:", device)


# ==================================================
# Load feature files
# ==================================================

print()
print("Loading training features...")

train_data = torch.load(
    TRAIN_FILE,
    map_location="cpu"
)

validation_data = torch.load(
    VALIDATION_FILE,
    map_location="cpu"
)

test_data = torch.load(
    TEST_FILE,
    map_location="cpu"
)


X_train = train_data["features"]

y_train = train_data["labels"]


X_validation = validation_data["features"]

y_validation = validation_data["labels"]


X_test = test_data["features"]

y_test = test_data["labels"]


print()
print("Training shape:")
print(X_train.shape)

print()
print("Validation shape:")
print(X_validation.shape)

print()
print("Test shape:")
print(X_test.shape)


# ==================================================
# Create datasets
# ==================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

validation_dataset = TensorDataset(
    X_validation,
    y_validation
)

test_dataset = TensorDataset(
    X_test,
    y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==================================================
# LSTM Model
# ==================================================

class EventLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout
    ):

        super().__init__()


        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )


        self.dropout = nn.Dropout(
            dropout
        )


        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        output, _ = self.lstm(x)


        # Take the final time step

        last_output = output[:, -1, :]


        last_output = self.dropout(
            last_output
        )


        output = self.classifier(
            last_output
        )


        return output


# ==================================================
# Create model
# ==================================================

model = EventLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)


model = model.to(device)


print()
print("Model:")
print(model)


# ==================================================
# Class weights
# ==================================================

# Count samples for each class

class_counts = torch.bincount(
    y_train,
    minlength=NUM_CLASSES
).float()


print()
print("Class counts:")
print(class_counts)


# Avoid division by zero

class_counts[class_counts == 0] = 1


class_weights = (
    len(y_train) /
    (
        NUM_CLASSES *
        class_counts
    )
)


class_weights = class_weights.to(
    device
)


print()
print("Class weights:")
print(class_weights)


# ==================================================
# Loss function
# ==================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==================================================
# Optimizer
# ==================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# Training
# ==================================================

best_validation_accuracy = 0.0


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    model.train()


    total_loss = 0.0

    correct = 0

    total = 0


    for features, labels in train_loader:

        features = features.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            features
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        total_loss += (
            loss.item() *
            labels.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    train_loss = (
        total_loss /
        total
    )


    train_accuracy = (
        correct /
        total
    )


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()


    validation_correct = 0

    validation_total = 0

    validation_loss_total = 0.0


    with torch.no_grad():

        for features, labels in validation_loader:

            features = features.to(device)

            labels = labels.to(device)


            outputs = model(
                features
            )


            loss = criterion(
                outputs,
                labels
            )


            validation_loss_total += (
                loss.item() *
                labels.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            validation_correct += (
                predictions == labels
            ).sum().item()


            validation_total += (
                labels.size(0)
            )


    validation_loss = (
        validation_loss_total /
        validation_total
    )


    validation_accuracy = (
        validation_correct /
        validation_total
    )


    print()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.4f}"
    )


    # ----------------------------------------------
    # Save best model
    # ----------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    INPUT_SIZE,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "num_classes":
                    NUM_CLASSES,

                "dropout":
                    DROPOUT,

                "classes": [
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
            },
            BEST_MODEL
        )


        print(
            "Best model saved."
        )


# ==================================================
# Load best model
# ==================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


checkpoint = torch.load(
    BEST_MODEL,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


# ==================================================
# Test
# ==================================================

test_correct = 0

test_total = 0


with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)

        labels = labels.to(device)


        outputs = model(
            features
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_correct += (
            predictions == labels
        ).sum().item()


        test_total += (
            labels.size(0)
        )


test_accuracy = (
    test_correct /
    test_total
)


print()
print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)


print()
print("Best validation accuracy:")

print(
    f"{best_validation_accuracy:.4f}"
)


print()
print("Model saved at:")

print(BEST_MODEL)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ==============================
# SETTINGS
# ==============================

FEATURE_PATH = r"E:\SprotsHighlights\datasets\swin_features\test_features.pt"
MODEL_PATH = r"E:\SprotsHighlights\models\swin_lstm_event_model.pth"

NUM_CLASSES = 15
INPUT_SIZE = 768
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

CLASS_NAMES = [
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


# ==============================
# MODEL
# ==============================

class EventLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_SIZE,
            NUM_CLASSES
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        result = self.classifier(last_output)

        return result


# ==============================
# DEVICE
# ==============================

device = torch.device("cpu")

print("Device:", device)


# ==============================
# LOAD TEST FEATURES
# ==============================

print("\nLoading test features...")

data = torch.load(
    FEATURE_PATH,
    map_location=device
)

X_test = data["features"]
y_test = data["labels"]

print("Test feature shape:")
print(X_test.shape)

print("Test label shape:")
print(y_test.shape)


# ==============================
# LOAD MODEL
# ==============================

print("\nLoading trained model...")

model = EventLSTM()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()

print("Model loaded successfully.")


# ==============================
# PREDICTION
# ==============================

print("\nGenerating predictions...")

all_predictions = []

batch_size = 16

with torch.no_grad():

    for i in range(0, len(X_test), batch_size):

        batch = X_test[i:i + batch_size]

        batch = batch.to(device)

        outputs = model(batch)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test.numpy(),
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ==============================
# CONFUSION MATRIX
# ==============================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

matrix = confusion_matrix(
    y_test.numpy(),
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(matrix)


# ==============================
# TEST ACCURACY
# ==============================

correct = 0

for i in range(len(y_test)):

    if y_test[i].item() == all_predictions[i]:
        correct += 1

accuracy = correct / len(y_test)

print("\n")
print("=" * 70)
print("FINAL TEST ACCURACY")
print("=" * 70)

print(f"{accuracy * 100:.2f}%")


# ==============================
# PREDICTED CLASS DISTRIBUTION
# ==============================

print("\n")
print("=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)

prediction_counts = Counter(all_predictions)

for i in range(NUM_CLASSES):

    count = prediction_counts.get(i, 0)

    print(
        f"{CLASS_NAMES[i]:25s}: {count}"
    )


print("\nEvaluation complete.")

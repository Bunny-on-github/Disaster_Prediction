import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# -------- Step 0: Load Preprocessed Data --------
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# y_train and y_test might be DataFrames — convert to Series if needed
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

# -------- Step 1: Initialize & Train the Model --------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    random_state=42,
    class_weight='balanced_subsample'
)
model.fit(X_train, y_train)

# -------- Step 2: Evaluate the Model --------
y_pred = model.predict(X_test)

print("✅ Model Training Complete\n")
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------- Step 3: Save the Trained Model --------
joblib.dump(model, "disaster_prediction_model.pkl")
print("\n💾 Model saved as 'disaster_prediction_model.pkl'")

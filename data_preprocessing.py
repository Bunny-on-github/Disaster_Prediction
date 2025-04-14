import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv(r"C:\Users\Yuvraj\Downloads\VS Code\GC_FL-ML\synthetic_natural_disasters.csv")

# -------- Step 1: Basic Cleaning --------
# Strip column names and drop duplicates
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.drop_duplicates(inplace=True)

# Handle missing values - drop rows with any missing values (customize as needed)
df.dropna(inplace=True)

# -------- Step 2: Feature/Target Split --------
# Define the target column (change if different)
target_col = 'disaster_type'

# Drop non-predictive identifiers if any (like ID, event name)
drop_cols = ['event_id', 'event_name', 'country', 'timestamp'] if set(['event_id', 'event_name', 'timestamp']).issubset(df.columns) else []
X = df.drop(columns=drop_cols + [target_col])
y = df[target_col]

# -------- Step 3: Encode Categorical Features --------
# Encode categorical columns
categorical_cols = X.select_dtypes(include='object').columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # save encoder for future use

# -------- Step 4: Normalize Numeric Features --------
numeric_cols = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# -------- Step 5: Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- Step 6: Display Basic Stats --------
print("✅ Preprocessing Complete.")
print(f"Features shape: {X.shape}")
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print("Class distribution:\n", y.value_counts())

# Optionally save processed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

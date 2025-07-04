import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

print("=== MACHINE LEARNING TUTORIAL: PERSONALITY PREDICTION ===\n")

# STEP 1: DATA LOADING AND EXPLORATION
print("STEP 1: Loading and Exploring Data")
print("-" * 40)

dataset = pd.read_csv("csv/personality_dataset.csv")
print(f"Dataset shape: {dataset.shape}")
print(f"Features: {list(dataset.columns)}")
print("\nFirst 5 rows:")
print(dataset.head())

print("\nDataset info:")
print(dataset.info())

print("\nMissing values:")
print(dataset.isnull().sum())

print("\nTarget distribution:")
print(dataset['Personality'].value_counts())

# STEP 2: DATA PREPROCESSING
print("\n\nSTEP 2: Data Preprocessing")
print("-" * 40)

# Handle missing values first
print("Handling missing values...")
# For numerical columns, use median imputation
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
imputer = SimpleImputer(strategy='median')
dataset[numerical_cols] = imputer.fit_transform(dataset[numerical_cols])

# Convert categorical to numerical (Label Encoding)
print("Converting categorical variables to numerical...")

# Extroverts = 1, Introvert = 0
dataset["Personality"] = (dataset["Personality"] == "Extrovert").astype(int)

# Yes = 1, No = 0
dataset["Stage_fear"] = (dataset["Stage_fear"] == "Yes").astype(int)
dataset["Drained_after_socializing"] = (dataset["Drained_after_socializing"] == "Yes").astype(int)

print("Data after preprocessing:")
print(dataset.head())
print(f"\nNo missing values: {dataset.isnull().sum().sum() == 0}")

# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
print("\n\nSTEP 3: Exploratory Data Analysis")
print("-" * 40)

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Feature distributions by personality type
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
           'Going_outside', 'Drained_after_socializing', 'Friends_circle_size']

for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    dataset.boxplot(column=feature, by='Personality', ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Personality')
    axes[row, col].set_xlabel('Personality (0=Introvert, 1=Extrovert)')

plt.tight_layout()
plt.show()

# STEP 4: FEATURE PREPARATION
print("\n\nSTEP 4: Preparing Features and Target")
print("-" * 40)

# Separate features (X) and target (y)
X = dataset.drop('Personality', axis=1)  # Features
y = dataset['Personality']  # Target

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {list(X.columns)}")

# STEP 5: TRAIN-TEST SPLIT
print("\n\nSTEP 5: Splitting Data into Training and Testing Sets")
print("-" * 40)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training target distribution:\n{y_train.value_counts()}")
print(f"Testing target distribution:\n{y_test.value_counts()}")

# STEP 6: FEATURE SCALING
print("\n\nSTEP 6: Feature Scaling")
print("-" * 40)

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")
print(f"Original feature means: {X_train.mean().round(2).tolist()}")
print(f"Scaled feature means: {X_train_scaled.mean(axis=0).round(2).tolist()}")

# STEP 7: MODEL TRAINING AND EVALUATION
print("\n\nSTEP 7: Training Multiple Machine Learning Models")
print("-" * 40)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    if name == 'Support Vector Machine':
        # SVM works better with scaled data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=['Introvert', 'Extrovert']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Introvert', 'Extrovert'],
                yticklabels=['Introvert', 'Extrovert'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# STEP 8: MODEL COMPARISON
print("\n\nSTEP 8: Model Comparison")
print("-" * 40)

print("Model Performance Summary:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

best_model = max(results, key=results.get)
print(f"\nBest performing model: {best_model} with accuracy: {results[best_model]:.4f}")

# STEP 9: FEATURE IMPORTANCE (for Random Forest)
print("\n\nSTEP 9: Feature Importance Analysis")
print("-" * 40)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance (Random Forest):")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Predicting Personality Type')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# STEP 10: MAKING PREDICTIONS ON NEW DATA
print("\n\nSTEP 10: Making Predictions on New Data")
print("-" * 40)

# Example: Predict personality for a new person
new_person = pd.DataFrame({
    'Time_spent_Alone': [7.0],
    'Stage_fear': [1],  # Yes
    'Social_event_attendance': [2.0],
    'Going_outside': [3.0],
    'Drained_after_socializing': [1],  # Yes
    'Friends_circle_size': [3.0],
    'Post_frequency': [2.0]
})

prediction = rf_model.predict(new_person)
probability = rf_model.predict_proba(new_person)

personality = "Extrovert" if prediction[0] == 1 else "Introvert"
confidence = max(probability[0]) * 100

print(f"New person characteristics:")
print(new_person)
print(f"\nPredicted personality: {personality}")
print(f"Confidence: {confidence:.1f}%")

print("\n=== MACHINE LEARNING TUTORIAL COMPLETED ===")
print("\nKey Concepts Learned:")
print("1. Data Loading and Exploration")
print("2. Data Preprocessing (handling missing values, encoding)")
print("3. Exploratory Data Analysis (EDA)")
print("4. Feature-Target separation")
print("5. Train-Test Split")
print("6. Feature Scaling")
print("7. Multiple Model Training (Logistic Regression, Random Forest, SVM)")
print("8. Model Evaluation (Accuracy, Classification Report, Confusion Matrix)")
print("9. Model Comparison")
print("10. Feature Importance Analysis")
print("11. Making Predictions on New Data")

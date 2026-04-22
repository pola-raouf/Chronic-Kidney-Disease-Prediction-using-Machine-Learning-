import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin

path = "E:\\AI Project\\Datasets\\Chronic KIdney Disease dataset.csv"
df = pd.read_csv(path)

# Custom IQR Outlier Remover
class IQRRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.Q1_ = X.quantile(0.25)
        self.Q3_ = X.quantile(0.75)
        self.IQR_ = self.Q3_ - self.Q1_
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        lower = self.Q1_ - self.factor * self.IQR_
        upper = self.Q3_ + self.factor * self.IQR_

        # Clip instead of dropping rows (pipeline-safe)
        X = X.clip(lower=lower, upper=upper, axis=1)
        return X.values
    

# Function to plot boxplots for outliers
def plot_outliers_before_after(before_df, after_df, numeric_cols, title_before="Before Preprocessing", title_after="After Preprocessing"):

    plt.figure(figsize=(15, 6))
    
    # Before preprocessing
    plt.subplot(1, 2, 1)
    sns.boxplot(data=before_df[numeric_cols], orient='h')
    plt.title(title_before)
    plt.xlabel("Value")

    plt.subplot(1, 2, 2)

    cols_after = [col for col in numeric_cols if col in after_df.columns]
    sns.boxplot(data=after_df[cols_after], orient='h')
    plt.title(title_after)
    plt.xlabel("Value")
    
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(before_df, after_df, numeric_cols, bins=15):

    # Before preprocessing
    before_df[numeric_cols].hist(bins=bins, figsize=(15, 10))
    plt.suptitle("Numerical Feature Distributions - Before Preprocessing", fontsize=16)
    plt.show()
    
    # After preprocessing
    cols_after = [col for col in numeric_cols if col in after_df.columns]
    after_df[cols_after].hist(bins=bins, figsize=(15, 10))
    plt.suptitle("Numerical Feature Distributions - After Preprocessing", fontsize=16)
    plt.show()

def plot_smote_before_after(X_train, y_train, preprocessor, random_state=42):

    # Class distribution BEFORE SMOTE
    plt.figure(figsize=(6, 4))
    y_train.value_counts().plot(kind='bar')
    plt.title("Class Distribution Before SMOTE (Training Set)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X_train_processed, y_train)

    # Class distribution AFTER SMOTE
    plt.figure(figsize=(6, 4))
    pd.Series(y_smote).value_counts().plot(kind='bar')
    plt.title("Class Distribution After SMOTE (Training Set)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def compute_cv_metrics(model_pipeline, X_train, y_train, cv):
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model_pipeline.fit(X_tr, y_tr)
        y_pred = model_pipeline.predict(X_val)
        
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred))
        metrics['recall'].append(recall_score(y_val, y_pred))
        metrics['f1'].append(f1_score(y_val, y_pred))
        
    # Compute mean ± std
    metrics_summary = {}
    for key, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        metrics_summary[key] = f"{mean:.6f} ± {std:.6f}"

    return metrics_summary

def plot_model_comparison(model_names, mean_accuracies, title="Model Comparison Using Cross-Validation"):

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, mean_accuracies)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Mean CV Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_auc(y_true, y_probs, model_name="Model"):
    # Compute ROC AUC
    roc_auc = roc_auc_score(y_true, y_probs)
    print(f"{model_name} ROC AUC: {roc_auc:.4f}")

    # Compute ROC curve points
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    # Plot ROC curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', label=f'{model_name} ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Find columns that contain "?" values
cols_with_question_mark = [col for col in df.columns if (df[col] == "?").any()]

print("Columns that contain '?':")
for col in cols_with_question_mark:
    print(f"- {col}")

# Find columns that have spaces in their names
cols_with_spaces = [col for col in df.columns if " " in col]

print("Columns that contain spaces:")
for col in cols_with_spaces:
    print(f"- {col}")

# Cleaning
df.replace("?", np.nan, inplace=True)
df.columns = df.columns.str.strip().str.replace(" ", "_")
df.drop(columns=["id"], inplace=True)
df["classification"] = df["classification"].str.strip()

X = df.drop("classification", axis=1)
y = df["classification"].map({"notckd": 0, "ckd": 1})

X_before_preprocessing = X.copy()

X_before_preprocessing.to_excel(
    r"E:\AI Project\Excel Sheets\D1\X_Before_Preprocessing.xlsx",
    index=False
)

print("X before preprocessing saved to X_Before_Preprocessing.xlsx")

X_before_3 = X.head(3)

X_before_3.to_excel(
    r"E:\AI Project\Excel Sheets\D1\X_First3_Before_Preprocessing.xlsx",
    index=False
)

print("First 3 rows BEFORE preprocessing saved.")

# Feature types
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

num_labels = df["classification"].nunique()  # defines the variable

print("Number of numerical features:", len(num_features))
print("Number of categorical features:", len(cat_features))
print("Number of class labels:", num_labels)
print("Class labels:", df["classification"].unique())

num_records = df.shape[0]
print("Number of records:", num_records)

print("\n=== Numeric Columns Summary ===")
print(X[num_features].describe())

# Numerical missing value analysis
missing_threshold = 0.5

num_missing_ratio = X[num_features].isna().mean()

num_drop_cols = num_missing_ratio[num_missing_ratio >= missing_threshold].index.tolist()
num_keep_cols = num_missing_ratio[num_missing_ratio < missing_threshold].index.tolist()

print("\nNumerical columns missing-value analysis:")
for col in num_missing_ratio.index:
    ratio = num_missing_ratio[col]
    if col in num_drop_cols:
        print(f"{col}: {ratio:.2%} missing -> DROPPED")
    else:
        print(f"{col}: {ratio:.2%} missing -> KNN IMPUTATION")

X = X.drop(columns=num_drop_cols)

num_features = num_keep_cols

# Detecting cols that need (Label Hot Encoder/ One Hot Encoder)
label_encode_cols = []
onehot_encode_cols = []

print("\nCategorical columns encoding suggestion:")

for col in cat_features:
    n_unique = X[col].nunique()
    if n_unique == 2:
        onehot_encode_cols.append(col)
        print(f"- {col} ({n_unique} unique values) → One-Hot Encoding (0/1)")
    else:
        label_encode_cols.append(col)
        print(f"- {col} ({n_unique} unique values) → Label Encoding (0/1/2/..)")

print("\nSummary:")
print("Columns for Label Encoding:", label_encode_cols)
print("Columns for One-Hot Encoding:", onehot_encode_cols)

# Numerical Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('iqr', IQRRemover(factor=1.5)),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

label_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Build preprocessor dynamically
transformers = [('num', numeric_transformer, num_features)]

if onehot_encode_cols:
    transformers.append(('onehot', onehot_transformer, onehot_encode_cols))

if label_encode_cols:
    transformers.append(('label', label_transformer, label_encode_cols))

preprocessor = ColumnTransformer(transformers=transformers)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Cross-validation loop
model_names = []
mean_accuracies = []
cv_results = {}
train_metrics_all_models = {}

for name, model in models.items():
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    metrics_summary = compute_cv_metrics(pipeline, X_train, y_train, cv)
    train_metrics_all_models[name] = metrics_summary

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    model_names.append(name)
    mean_accuracies.append(scores.mean())

    cv_results[name] = list(scores) + [scores.mean()]

    print(f"\n{name}")
    print("CV scores:", scores)
    print("Mean CV accuracy:", scores.mean())

# Create CV results table
fold_columns = [f"Fold {i+1}" for i in range(cv.get_n_splits())] + ["Mean Accuracy"]

cv_df = pd.DataFrame(cv_results, index=fold_columns).transpose()

# Save to Excel
cv_df.to_excel(r"E:\AI Project\Excel Sheets\D1\CV_Results.xlsx", index=True)

# Convert to DataFrame for a nice table
train_metrics_df = pd.DataFrame(train_metrics_all_models).transpose()
train_metrics_df.index.name = "Model"
train_metrics_df.reset_index(inplace=True)

# Save to Excel
train_metrics_df.to_excel(r"E:\AI Project\Excel Sheets\D1\Train_Metrics_CV.xlsx", index=False)
print("Training set metrics (mean ± std) saved to Train_Metrics_CV.xlsx")
print(train_metrics_df)

print("\nCross-validation results saved to CV_Results.xlsx")
print(cv_df)

# Final model training
final_model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Generate classification report as dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Optional: round values for nicer presentation
report_df = report_df.round(4)
report_df.to_excel(r"E:\AI Project\Excel Sheets\D1\Classification_Report.xlsx", index=True)
print("Classification report saved to Classification_Report.xlsx")

print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", report_df)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Train vs Test comparison
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\nTrain vs Test Performance")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Gap (Train - Test): {(train_acc - test_acc):.4f}")

# Transformation Independently (Testing to see the difference between the features before and after)
# Learning parameters and transforming it on the X_train
X_train_preprocessed = preprocessor.fit_transform(X_train)
if 'onehot' in preprocessor.named_transformers_:
    onehot_columns = preprocessor.named_transformers_['onehot'].named_steps['onehot'].get_feature_names_out(onehot_encode_cols)
else:
    onehot_columns = []
all_columns = list(num_features) + list(onehot_columns) + list(label_encode_cols)
X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=all_columns)

X_train_preprocessed_df.copy()

X_train_preprocessed_df.to_excel(
    r"E:\AI Project\Excel Sheets\D1\X_After_Preprocessing.xlsx",
    index=False)

X_after_3 = X_train_preprocessed_df.head(3)

X_after_3.to_excel(
    r"E:\AI Project\Excel Sheets\D1\X_First3_After_Preprocessing.xlsx",
    index=False
)

print("First 3 rows AFTER preprocessing saved.")

# Outliers Imputation without scaling (Testing) INDEPENDENT
numeric_imputer = KNNImputer(n_neighbors=5)
X_train_num_imputed = pd.DataFrame(
    numeric_imputer.fit_transform(X_train[num_features]),
    columns=num_features,
    index=X_train.index   # keeps alignment
)

iqr_remover = IQRRemover(factor=1.5)
X_train_num_iqr = pd.DataFrame(
    iqr_remover.fit_transform(X_train_num_imputed),
    columns=num_features,
    index=X_train.index
)

# Show the first 40 rows of feature columns
print(X.head(40))

# Show first 40 rows
print(X_train_preprocessed_df.head(40))

plot_outliers_before_after(X, X_train_preprocessed_df, num_features)

# Outliers (Imputation, No Scaling)
plot_outliers_before_after(
    before_df=X,
    after_df=X_train_num_iqr,
    numeric_cols=num_features,
    title_before="Before Preprocessing",
    title_after="After Imputation (No Scaling)"
)

# Call the function to plot numeric distributions before and after preprocessing
plot_numeric_distributions(
    before_df=X,
    after_df=X_train_preprocessed_df,
    numeric_cols=num_features
)

plot_smote_before_after(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessor
)

plot_model_comparison(model_names, mean_accuracies)

# ROC AUC for all models
y_probs_rf = final_model.predict_proba(X_test)[:, 1]

plot_roc_auc(y_test, y_probs_rf, model_name="Random Forest")

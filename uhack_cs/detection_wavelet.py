# !pip install PyWavelets
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Function for encoding categorical columns
def encode_categorical_columns(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

# Function for applying Wavelet Transform
def apply_wavelet_transform(X, wavelet='db1', level=1):
    transformed_data = []

    # Apply wavelet transform to each feature column in X
    for col in X.columns:
        data = X[col].values
        coeffs = pywt.wavedec(data, wavelet, level=level)  # Decompose data into wavelet coefficients
        transformed_col = np.concatenate(coeffs)  # Concatenate coefficients to form transformed feature
        transformed_data.append(transformed_col)

    # Convert list of transformed columns back to a DataFrame
    transformed_df = pd.DataFrame(np.array(transformed_data).T, columns=X.columns)
    return transformed_df

# Function for standardizing features
def standardize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# Full Data Preprocessing Pipeline
def preprocess_data(df, wavelet='db1', wavelet_level=1):
    # Step 1: Encode categorical columns
    df_encoded = encode_categorical_columns(df)

    # Step 2: Apply wavelet transform to the encoded data (for numerical columns)
    transformed_df = apply_wavelet_transform(df_encoded, wavelet, wavelet_level)

    # Step 3: Standardize the data
    standardized_df = standardize_data(transformed_df)

    return standardized_df

# Example usage
# df = pd.read_csv('your_dataset.csv')
# preprocessed_data = preprocess_data(df, wavelet='db1', wavelet_level=1)


df = pd.read_csv('/wavelet_dataset.csv')  # Load your dataset
preprocessed_data = preprocess_data(df, wavelet='db1', wavelet_level=1)  # Preprocess the data


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Function to apply PCA for feature projection
def apply_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

# Function to apply LDA for feature projection
def apply_lda(X, y, n_components=None):
    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    return X_lda, lda

# Full Feature Projection Pipeline
def feature_projection(X, y, method='pca', n_components=None):
    if method == 'pca':
        X_projected, model = apply_pca(X, n_components)
        print(f'PCA: {n_components} components selected')
    elif method == 'lda':
        X_projected, model = apply_lda(X, y, n_components)
        print(f'LDA: {n_components} components selected')
    else:
        raise ValueError("Method should be 'pca' or 'lda'")

    return X_projected, model

# Example usage
# df = pd.read_csv('your_dataset.csv')
# Preprocess data first (use the preprocessing steps from before)
# X = preprocessed_data
# y = target_column

# Apply PCA with 5 components
# X_pca, pca_model = feature_projection(X, y=None, method='pca', n_components=5)

# Apply LDA with 2 components
# X_lda, lda_model = feature_projection(X, y=target_column, method='lda', n_components=2)


import pandas as pd
import pywt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ... (Your existing functions: encode_categorical_columns, apply_wavelet_transform, standardize_data, preprocess_data, apply_pca, apply_lda, feature_projection remain the same) ...

df = pd.read_csv('/WSN-DS.csv')  # Load your dataset

# Get the target column from the original DataFrame before preprocessing
target_column = df['Attack type']  # Replace 'target' with your actual target column name

# Ensure target_column and preprocessed data have the same length:
preprocessed_data = preprocess_data(df, wavelet='db1', wavelet_level=1)  # Preprocess the data

# Align the lengths of preprocessed_data and target_column
min_len = min(preprocessed_data.shape[0], target_column.shape[0])
preprocessed_data = preprocessed_data[:min_len]
target_column = target_column[:min_len]


# Now you can use target_column for LDA:
X_lda, lda_model = feature_projection(preprocessed_data, y=target_column, method='lda', n_components=2)

X_pca, pca_model = feature_projection(preprocessed_data, y=None, method='pca', n_components=5)




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


# Example function to train and predict using a weighted ensemble
def weighted_ensemble(X_train, X_test, y_train, y_test, weights):
    # Initialize the models
    lr_model = LogisticRegression()
    rf_model = RandomForestClassifier()
    svm_model = SVC(probability=True)
    gb_model = GradientBoostingClassifier()
    knn_model = KNeighborsClassifier()

    # Fit the models on the training data
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    # Make predictions (probabilities) for each model
    lr_preds = lr_model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class
    rf_preds = rf_model.predict_proba(X_test)[:, 1]
    svm_preds = svm_model.predict_proba(X_test)[:, 1]
    gb_preds = gb_model.predict_proba(X_test)[:, 1]
    knn_preds = knn_model.predict_proba(X_test)[:, 1]

    # Combine predictions using the weighted sum
    final_preds = (weights['a'] * lr_preds +
                   weights['b'] * rf_preds +
                   weights['c'] * svm_preds +
                   weights['d'] * gb_preds +
                   weights['e'] * knn_preds)

    # Convert probabilities to binary predictions (0 or 1)
    final_preds_binary = (final_preds > 0.5).astype(int)

    # Evaluate the model
    accuracy = accuracy_score(y_test, final_preds_binary)
    f1 = f1_score(y_test, final_preds_binary)

    print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Ensemble Model F1 Score: {f1:.2f}")

    return final_preds_binary, accuracy, f1

# Usage Example
if __name__ == "__main__":
    # Create a synthetic dataset (replace this with your dataset)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the weights for the ensemble (a, b, c, d, e)
    weights = {
        'a': 0.11,  # Logistic Regression
        'b': 0.41,  # Random Forest
        'c': 0.23,  # SVM
        'd': 0.13,  # Gradient Boosting
        'e': 0.12   # KNN
    }

    # Call the ensemble function
    predictions, ensemble_accuracy, ensemble_f1 = weighted_ensemble(X_train, X_test, y_train, y_test, weights)


# !pip install matplotlib seaborn scikit-learn



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Function to output classification results
def output_classification_results(y_true, y_pred):
    # Creating a confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Displaying the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC Curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# Main function to integrate everything
if __name__ == "__main__":
    # Example dataset (use preprocessed dataset instead)
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # Create a synthetic dataset (replace this with your dataset)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the weights for the ensemble (a, b, c, d, e)
    weights = {
       'a': 0.11,  # Logistic Regression
        'b': 0.41,  # Random Forest
        'c': 0.23,  # SVM
        'd': 0.13,  # Gradient Boosting
        'e': 0.12   # KNN
    }

    # Call the ensemble function
    predictions, ensemble_accuracy = weighted_ensemble(X_train, X_test, y_train, y_test, weights)

    # Output the classification results
    output_classification_results(y_test, predictions)

    # Additional information about the IDS
    print("Classification results passed to IDS: ")
    print(predictions)

    print("Ensemble Model Accuracy: {:.2f}%".format(ensemble_accuracy * 100))





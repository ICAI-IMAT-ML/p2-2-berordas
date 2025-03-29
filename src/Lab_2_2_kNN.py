# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns
import pandas as pd


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """

    distancia = 0
    for i in range(len(a)):
        distancia += (abs(a[i] - b[i]) ** p) 

    distancia = distancia ** (1 / p)

    return distancia


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if len(X_train) != len(y_train):
            raise ValueError("Length of X_train and y_train must be equal.")
        
        if not isinstance(k, int) or not isinstance(p, int) or k <= 0 or p <= 0:
            raise ValueError("k and p must be positive integers.")
    
        self.k = k
        self.p = p
        self.x_train = X_train
        self.y_train = y_train

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        prediccion = []
        for i in range(len(X)):
            distancias = self.compute_distances(X[i])
            filas_cercanas = self.get_k_nearest_neighbors(distancias)
            etiquetas = []
            for indice in filas_cercanas:
                etiquetas.append(self.y_train[indice])

            etiquetas = np.array(etiquetas)
            prediccion.append(self.most_common_label(etiquetas))

        return prediccion

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        prediccion = []
        for i in range(len(X)):
            distancias = self.compute_distances(X[i])
            filas_cercanas = self.get_k_nearest_neighbors(distancias)
            etiquetas = [self.y_train[indice] for indice in filas_cercanas] 

            etiquetas = np.array(etiquetas, dtype=str)

            verdaderos = np.sum(etiquetas == 'YES')
            falsos = np.sum(etiquetas == 'NO')

            total = verdaderos + falsos
            if total > 0:
                prediccion.append([falsos / total, verdaderos / total])
            else:
                prediccion.append([0.5, 0.5])  

        return np.array(prediccion)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        distancias = []
        for i in range(len(self.x_train)):
            distancia = minkowski_distance(point, self.x_train[i], self.p)
            distancias.append(distancia)

        return distancias

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        distancias_ordenadas = np.argsort(distances)
        k_filas_cercanas = distancias_ordenadas[:self.k]

        return k_filas_cercanas

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        contador = np.unique(knn_labels, return_counts=True)

        return contador[0][np.argmax(contador[1])]

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}
    # Predict on input data
    preds = np.array(model.predict(X))
    print(preds)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color='green', label=f'True {num_to_label[1]}')
    ax[0].scatter(X[fp, 0], X[fp, 1], color='red', label=f'False {num_to_label[1]}')
    ax[0].scatter(X[fn, 0], X[fn, 1], color='blue', label=f'False {num_to_label[0]}')
    ax[0].scatter(X[tn, 0], X[tn, 1], color='orange', label=f'True {num_to_label[0]}')
    ax[0].set_title('Classification Results')
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points_n),
                         np.linspace(y_min, y_max, grid_points_n))

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='Set1', ax=ax[1])
    ax[1].set_title('Classes and Estimated Probability Contour Lines')

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors='black')
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()

def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tp = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    tn = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
    fp = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    fn = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }

def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    y_verdaderos = np.array([1 if etiqueta == positive_label else 0 for etiqueta in y_true])

    df = pd.DataFrame({'y_true': y_verdaderos, 'y_probs': y_probs})

    bins = np.linspace(0, 1, n_bins + 1)
    df['bins'] = pd.cut(df['y_probs'], bins=bins, include_lowest=True)

    grouped = df.groupby('bins').agg({'y_true': 'mean', 'y_probs': 'mean'}).reset_index()
    
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    true_proportions = grouped['y_true'].values

    plt.figure(figsize=(8, 8))
    plt.plot(grouped['y_probs'], true_proportions, marker='*', linestyle='-', label='Calibración')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideal')
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Proporción de positivos')
    plt.title('Curva de Calibración')
    plt.legend()
    plt.show()

    return {"bin_centers": bin_centers, "true_proportions": true_proportions}

def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    y_true_mapped = np.array([1 if etiqueta == positive_label else 0 for etiqueta in y_true])

    df = pd.DataFrame({'y_true': y_true_mapped, 'y_probs': y_probs})

    positive_probs = df[df['y_true'] == 1]['y_probs']
    negative_probs = df[df['y_true'] == 0]['y_probs']

    plt.figure(figsize=(10, 6))
    plt.hist([positive_probs, negative_probs], bins=n_bins, alpha=0.5, label=['Positivos Verdaderos', 'Falsos Positivos'])
    plt.legend()
    plt.title('Distribución de Probabilidades - Positivos')
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Frecuencia')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist([negative_probs, positive_probs], bins=n_bins, alpha=0.5, label=['Negativos Verdaderos', 'Falsos Negativos'])
    plt.legend()
    plt.title('Distribución de Probabilidades - Negativos')
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Frecuencia')
    plt.show()

    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }

def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """
    y_true = np.array([1 if etiqueta == positive_label else 0 for etiqueta in y_true])
    tpr_list = []
    fpr_list = []

    thresholds = np.linspace(0, 1, 11)
    for idx, thresh in enumerate(thresholds):
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if tp + fn != 0 else 0
        fpr = fp / (fp + tn) if fp + tn != 0 else 0

        tpr_list.insert(0, tpr)
        fpr_list.insert(0, fpr)

    plt.plot(fpr_list, tpr_list, label='Curva ROC')
    plt.plot([0, 1], [0, 1])
    plt.xlabel('Falsos positivos')
    plt.ylabel('Verdaderos positivos')
    plt.legend()
    plt.show()

    return {"fpr": np.array(fpr_list), "tpr": np.array(tpr_list)}

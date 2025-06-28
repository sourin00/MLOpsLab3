
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def load_and_prepare_data():

    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return digits, data, digits.target


def create_and_train_classifier(X_train, y_train, gamma=0.001):

    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)
    return clf


def split_data(data, target, test_size=0.5, shuffle=False):

    return train_test_split(data, target, test_size=test_size, shuffle=shuffle)


def visualize_samples(images, labels, title_prefix="Sample", n_samples=4, figsize=(10, 3)):

    _, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=figsize)
    
    for i, (ax, image, label) in enumerate(zip(axes, images[:n_samples], labels[:n_samples])):
        ax.set_axis_off()
        
        # Reshape if image is flattened
        if image.ndim == 1:
            image = image.reshape(8, 8)
            
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"{title_prefix}: {label}")


def print_classification_report(classifier, y_test, predicted):

    print(f"Classification report for classifier {classifier}:\n"
          f"{metrics.classification_report(y_test, predicted)}\n")


def create_confusion_matrix(y_test, predicted):

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp


def rebuild_classification_report_from_confusion_matrix(confusion_matrix):

    y_true = []
    y_pred = []
    
    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(confusion_matrix)):
        for pred in range(len(confusion_matrix)):
            y_true += [gt] * confusion_matrix[gt][pred]
            y_pred += [pred] * confusion_matrix[gt][pred]
    
    print("Classification report rebuilt from confusion matrix:\n"
          f"{metrics.classification_report(y_true, y_pred)}\n")


def run_complete_classification(gamma=0.001, test_size=0.5, shuffle=False):


    digits, data, target = load_and_prepare_data()
    

    X_train, X_test, y_train, y_test = split_data(data, target, test_size, shuffle)
    

    clf = create_and_train_classifier(X_train, y_train, gamma)
    

    predicted = clf.predict(X_test)
    

    visualize_samples(digits.images, digits.target, "Training")
    

    visualize_samples(X_test, predicted, "Prediction")
    

    print_classification_report(clf, y_test, predicted)
    

    disp = create_confusion_matrix(y_test, predicted)
    

    rebuild_classification_report_from_confusion_matrix(disp.confusion_matrix)
    
    return clf, y_test, predicted, disp

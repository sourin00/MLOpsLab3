

import matplotlib.pyplot as plt
from utils import run_complete_classification

def main():


    classifier, y_test, predicted, confusion_matrix_display = run_complete_classification(
        gamma=0.001,
        test_size=0.5,
        shuffle=False
    )


    plt.show()


if __name__ == "__main__":
    main()
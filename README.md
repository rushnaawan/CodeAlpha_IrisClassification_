# Iris Flower Classification

Task 1 of my CodeAlpha Data Science internship.

The model predicts the species of an iris flower (setosa, versicolor, virginica) from its sepal and petal measurements. The dataset is `Iris.csv` (included in this folder).

## How to run
    pip install -r requirements.txt
    python iris_classification.py

Plots and results are saved in the outputs folder.

## What I did
- Loaded Iris.csv (150 rows, 4 features, no missing values, 50 flowers of each species, 3 duplicate rows which I kept)
- Made a pairplot to see how the species separate. Petal length and width separate them best.
- Split the data into 80% train and 20% test
- Tried Logistic Regression, KNN, Decision Tree, Random Forest and SVM
- Compared them using 5-fold cross validation on the training data and picked the best one from that, not from the test score
- Checked the chosen model on the test set with a classification report and a confusion matrix
- Predicted the species of one new flower (5.1, 3.5, 1.4, 0.2), which came out as setosa

## Results
| Model | CV accuracy | Test accuracy |
|---|---|---|
| Logistic Regression | 0.958 | 0.933 |
| KNN | 0.967 | 0.933 |
| Decision Tree | 0.942 | 0.933 |
| Random Forest | 0.950 | 0.900 |
| SVM | 0.967 | 0.967 |

KNN and SVM tied on cross validation, and the script takes the first one (KNN). KNN got 93% on the test set. Setosa was classified perfectly. The few mistakes were between versicolor and virginica, which also overlap in the pairplot.

The test set only has 30 rows, so one wrong prediction changes the accuracy by about 3%. Because of this, the test accuracies of the models are too close to say which one is really better.

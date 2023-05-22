import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import random
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
import time


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, fbeta_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier



labels = ['IT', 'ES', 'US', 'GE', 'FR', 'UK']

#import from csv
def import_establishments(log=False):
    # no missing values
    data = pd.read_csv(r'accent-mfcc-data-1.csv')
    if (log):
        #data size
        print(f"Data length: {len(data)}\n")

        #see how much info we have about each language
        print("Count -> group by language")
        print(data.groupby("language").count(), "\n")

        #see X1..X12 count/mean/std/min/max...
        print(data.describe())

        #see violin plots
        plt.figure(figsize=(8, 8))
        i = 1
        for column in data.columns:
            if column == 'language': continue
            plt.subplot(3, 4, i)
            sb.violinplot(x='language', y=column, data=data)
            i += 1
        plt.show()

    return data.values.tolist()

#------------------- Data split and pre-processing ---------------------

#make number of samples of each class equal
def balance_data(data: list):
    us_lines = []
    uk_lines = []
    rest = []

    for line in data:
        if line[0] == "US": us_lines.append(line)
        elif line[0] == "UK": uk_lines.append(line)
        else: rest.append(line)
    
    us_lines = random.sample(us_lines, 30)
    uk_lines = random.sample(uk_lines, 30)

    return rest + uk_lines + us_lines

#split data in 2 sets 
def split_data(data: list, test_percentage: float):
    #X values (without label)
    x_values = list(map(lambda x: x[1:], data))
    
    #labels
    labels = list(map(lambda x: x[0], data))

    result = train_test_split(x_values, labels, test_size=test_percentage)

    return {"train_x": result[0], 
            "test_x": result[1],
            "train_label": result[2],
            "test_label": result[3]}

#create folds out of data, useful for cross_validation
def k_fold(data: list, n_splits: int):
    #Cada row é usada para treinar todos os folds menos um, onde é usada para testar
    #Cada fold tem um conjunto de teste que dá saltitos (ex. 0 1 2 10 11 12 ... )

    #X values (without label)
    x_values = list(map(lambda x: x[1:], data))
    
    #labels
    labels = list(map(lambda x: x[0], data))

    cv = StratifiedKFold(n_splits=n_splits) #stratified to mantain balance

    split_indices = cv.split(x_values, labels) #gives rows indices (train, test)

    split_values = []
    for train, test in split_indices:
        split_values.append(
            {"train_x": [x_values[i] for i in train], 
            "test_x": [x_values[i] for i in test],
            "train_label": [labels[i] for i in train],
            "test_label": [labels[i] for i in test]
            })
    
    return split_values


#apply k_fold and get list of (real_label, predicted_label)
def cross_validation(data:list, n_splits: int, algorithm):
    k_splits = k_fold(data, n_splits)
    cross_results = []
    for dic in k_splits:
        real_prediction = (dic["test_label"],algorithm(dic))
        cross_results.append(real_prediction)

    return cross_results  

#calculate scores
def scores_cross_validation(cross_results: list, scoring_func):
    scores = []
    for real, prediction in cross_results:
        scores.append(scoring_func(real, prediction))
    
    return scores



#------------------- Classification Algortihms ---------------------
#decision tree algorithm
def decision_tree(split_data: dict, estimator):
    #Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
    # Create the classifier and train
    clf = DecisionTreeClassifier()
    clf = clf.fit(split_data["train_x"], split_data["train_label"])

    #Predict
    prediction = clf.predict(split_data["test_x"])
    
    return prediction

def decision_tree_with_times(split_data: dict, estimator):
    clf = estimator
    
    start_time = time.time()
    clf = clf.fit(split_data["train_x"], split_data["train_label"])
    training_time = time.time() - start_time
    
    start_time = time.time()
    prediction = clf.predict(split_data["test_x"])
    testing_time = time.time() - start_time
    
    return prediction, training_time, testing_time



#------------------- Evaluation ------------------------------
#confusion matrix
def calculate_confusion_matrix(real: list,prediction: list, log=False):
    c_matrix = confusion_matrix(real, prediction, labels=labels)
    if(log):
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=labels)
        disp.plot()
        plt.show()
    return c_matrix

#accuracy - return the fraction of correctly classified samples
#in percentage
def calculate_accuracy(real: list,prediction: list): 
    return accuracy_score(real, prediction, normalize =True)

#precision - return (well-classified with class A/ all classified with class A)
#is intuitively the ability of the classifier not to label as positive a sample that is negative
def calculate_precision(real: list,prediction: list):
    return precision_score(real, prediction, labels=labels, average='weighted')
    #average paremter defines how to combine each class precision:
        #None - the scores for each class are returned.
        #'micro' - Calculate metrics globally by counting the total true positives, false negatives and false positives.
        #'macro' - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        #'weighted' - Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.


#recall - return (well-classified with class A/ (well-classified with class A + A elements badly classified with other class)=all_the_true_A)
#is intuitively the ability of the classifier to find all the positive samples
def calculate_recall(real: list,prediction: list):
    return recall_score(real, prediction, labels=labels, average='weighted')
    #average paremter defines how to combine each class precision:
        #None - the scores for each class are returned.
        #'micro' - Calculate metrics globally by counting the total true positives, false negatives and false positives.
        #'macro' - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
        #'weighted' - Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

#f-score - return a weighted harmonic mean of the precision and recall
def calculate_f_score(real: list,prediction: list):
    #The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
    return fbeta_score(real, prediction, beta=1.0 ,labels=labels, average='weighted')
    #average='weighted' - it can result in an F-score that is not between precision and recall.

#------------------- Tuning the algorithms -> changing  parameters------------------------------
#Grid search
#The idea behind Grid Search is simple: explore a range of 
# parameters and find the best-performing parameter combination.
# Focus your search on the best range of parameters, 
# then repeat this process several times until the best parameters are discovered.

#Tune decision_tree
def tune_decision_tree(data: list, log=False):
    #X values (without label)
    x_values = list(map(lambda x: x[1:], data))
    #labels
    labels = list(map(lambda x: x[0], data))

    decision_tree_classifier = DecisionTreeClassifier()

    #Trying theses parameters options
    parameter_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
                      'splitter': ['best', 'random'],
                      'max_depth': list(range(1, 15)),
                      'max_features': list(range(1,13)) }

    grid_search = GridSearchCV(decision_tree_classifier, param_grid=parameter_grid, scoring='accuracy', cv=10)
    grid_search.fit(x_values, labels)
    
    if(log):
        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

    return grid_search.best_estimator_
    

#Testing functions - is not main anymore
if __name__ != '__main__':
    data = import_establishments(False)
    data = balance_data(data)

    '''
    #check classes/label dominance
    label_count_train = {}
    for label in split["train_label"]:
        label_count_train[label] = label_count_train.get(label, 0) +1
    print(label_count_train)
    split = split_data(data, 0.25)
    '''
    
    #My version
    cross_result = cross_validation(data, 10, decision_tree)
    cross_scores = scores_cross_validation(cross_result, calculate_accuracy)
    print("My cross scores: ", cross_scores)
    

    #Their version
    x_values = list(map(lambda x: x[1:], data)) #X values (without label)
    labels = list(map(lambda x: x[0], data)) #labels
    clf = DecisionTreeClassifier()
    their_cross_scores = cross_val_score(clf, x_values, labels, scoring='accuracy', cv=10)
    print("Their cross scores: ", their_cross_scores)


    #tuning
    new_estimator = tune_decision_tree(data, True)
    their_cross_scores = cross_val_score(new_estimator, x_values, labels, scoring='accuracy', cv=10)
    print("Tunned their cross scores: ", their_cross_scores)




def evaluate_model(predictions: list, split_data: dict):
    # Comparing predicted labels with actual labels
    accuracy = accuracy_score(split_data["test_label"], predictions)
    return accuracy
    

def knn(split_data: dict, n_neighbors=5):
    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Treina o classifier
    knn.fit(split_data["train_x"], split_data["train_label"])
    
    # Previsão
    prediction = knn.predict(split_data["test_x"])
    
    return prediction

def knn_with_times(split_data: dict, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    start_time = time.time()
    knn.fit(split_data["train_x"], split_data["train_label"])
    training_time = time.time() - start_time
    
    start_time = time.time()
    prediction = knn.predict(split_data["test_x"])
    testing_time = time.time() - start_time
    
    return prediction, training_time, testing_time

def svm_mod(split_data: dict):
    # SVM Classifier
    svc = svm.SVC()
    
    # Treina o classifier
    svc.fit(split_data["train_x"], split_data["train_label"])
    
    # Previsão
    prediction = svc.predict(split_data["test_x"])
    
    return prediction    


def svm_mod_with_times(split_data: dict):
    svc = svm.SVC()
    
    start_time = time.time()
    svc.fit(split_data["train_x"], split_data["train_label"])
    training_time = time.time() - start_time
    
    start_time = time.time()
    prediction = svc.predict(split_data["test_x"])
    testing_time = time.time() - start_time
    
    return prediction, training_time, testing_time


param_grid = {
    'hidden_layer_sizes': [(50, 50), (100, 100), (200, 100, 50,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'max_iter': [10000],
    'learning_rate': ['adaptive']
}

def neural_network(split_data, estimator, log=False):
    # Train the estimator
    estimator.fit(split_data["train_x"], split_data["train_label"])

    # Make predictions on the test data
    predictions = estimator.predict(split_data["test_x"])

    # Calculate evaluation metrics
    if(log):
        confusion_matrix = calculate_confusion_matrix(split_data["test_label"], predictions, log=False)
        accuracy = calculate_accuracy(split_data["test_label"], predictions)
        precision = calculate_precision(split_data["test_label"], predictions)
        recall = calculate_recall(split_data["test_label"], predictions)
        f_score = calculate_f_score(split_data["test_label"], predictions)

        # Print or store the evaluation metrics
        print("Confusion Matrix:")
        print(confusion_matrix)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-Score:", f_score)

    return predictions

def neural_network_with_times(split_data, estimator, log=False):
    start_time = time.time()
    estimator.fit(split_data["train_x"], split_data["train_label"])
    training_time = time.time() - start_time
    
    start_time = time.time()
    predictions = estimator.predict(split_data["test_x"])
    testing_time = time.time() - start_time
    
    if log:
        confusion_matrix = calculate_confusion_matrix(split_data["test_label"], predictions, log=False)
        accuracy = calculate_accuracy(split_data["test_label"], predictions)
        precision = calculate_precision(split_data["test_label"], predictions)
        recall = calculate_recall(split_data["test_label"], predictions)
        f_score = calculate_f_score(split_data["test_label"], predictions)
        
        print("Confusion Matrix:")
        print(confusion_matrix)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-Score:", f_score)
    
    return predictions, training_time, testing_time

#for neural network
def grid_search(data, param_grid, cv=10):
    # Extract features and labels from the data
    x_values = [sample[1:] for sample in data]
    labels = [sample[0] for sample in data]

    # Create the neural network classifier
    classifier = MLPClassifier()

    # Perform grid search
    grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(x_values, labels)

    # Retrieve the best estimator
    best_estimator = grid_search.best_estimator_

    # Print the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    return best_estimator


if __name__ == '__main__':
    data = import_establishments(log=False)
    data = balance_data(data)

    best_estimator_nn = grid_search(data, param_grid)
    best_estimator_svm = svm_mod
    best_estimator_knn = knn
    best_estimator_dt = tune_decision_tree(data, log=False)


    # Cross Validation
    k_splits = k_fold(data, 10)

    algorithms = ['Neural Network', 'SVM', 'KNN', 'Decision Tree']
    
    model_accuracies = {
        'Neural Network': [],
        'SVM': [],
        'KNN':[],
        'Decision Tree':[]
    }

    model_precisions = {
        'Neural Network': [],
        'SVM': [],
        'KNN':[],
        'Decision Tree':[]
    }

    model_recalls = {
        'Neural Network': [],
        'SVM': [],
        'KNN':[],
        'Decision Tree':[]
    }

    model_f_scores = {
        'Neural Network': [],
        'SVM': [],
        'KNN':[],
        'Decision Tree':[]
    }

    model_training_times = {
        'Neural Network': [],
        'SVM': [],
        'KNN':[],
        'Decision Tree':[]
    }

    model_testing_times  = {
        'Neural Network': [],
        'SVM': [],
        'KNN':[],
        'Decision Tree':[]
    }


    for i, split in enumerate(k_splits):
        # Train and evaluate Neural Network
        nn_predictions, nn_training_time, nn_testing_time = neural_network_with_times(split, best_estimator_nn)
        
        nn_accuracy = evaluate_model(nn_predictions, split)
        nn_precision = calculate_precision(split["test_label"], nn_predictions)
        nn_recall = calculate_recall(split["test_label"], nn_predictions)
        nn_f_score = calculate_f_score(split["test_label"], nn_predictions)

        model_accuracies['Neural Network'].append(nn_accuracy)
        model_precisions['Neural Network'].append(nn_precision)
        model_recalls['Neural Network'].append(nn_recall)
        model_f_scores['Neural Network'].append(nn_f_score)
        model_training_times['Neural Network'].append(nn_training_time)
        model_testing_times['Neural Network'].append(nn_testing_time)

        # Train and evaluate SVM
        svm_predictions, svm_training_time, svm_testing_time = svm_mod_with_times(split)
        
        svm_accuracy = evaluate_model(svm_predictions, split)
        svm_precision = calculate_precision(split["test_label"], svm_predictions)
        svm_recall = calculate_recall(split["test_label"], svm_predictions)
        svm_f_score = calculate_f_score(split["test_label"], svm_predictions)

        model_accuracies['SVM'].append(svm_accuracy)
        model_precisions['SVM'].append(svm_precision)
        model_recalls['SVM'].append(svm_recall)
        model_f_scores['SVM'].append(svm_f_score)
        model_training_times['SVM'].append(svm_training_time)
        model_testing_times['SVM'].append(svm_testing_time)


        # Train and evaluate KNN
        knn_predictions, knn_training_time, knn_testing_time = knn_with_times(split)

        knn_accuracy = evaluate_model(knn_predictions, split)
        knn_precision = calculate_precision(split["test_label"], knn_predictions)
        knn_recall = calculate_recall(split["test_label"], knn_predictions)
        knn_f_score = calculate_f_score(split["test_label"], knn_predictions)

        model_accuracies['KNN'].append(knn_accuracy)
        model_precisions['KNN'].append(knn_precision)
        model_recalls['KNN'].append(knn_recall)
        model_f_scores['KNN'].append(knn_f_score)
        model_training_times['KNN'].append(knn_training_time)
        model_testing_times['KNN'].append(knn_testing_time)

        # Train and evaluate Decision Tree
        dt_predictions, dt_training_time, dt_testing_time = decision_tree_with_times(split, best_estimator_dt)
        
        dt_accuracy = evaluate_model(dt_predictions, split)
        dt_precision = calculate_precision(split["test_label"], dt_predictions)
        dt_recall = calculate_recall(split["test_label"], dt_predictions)
        dt_f_score = calculate_f_score(split["test_label"], dt_predictions)

        model_accuracies['Decision Tree'].append(dt_accuracy)
        model_precisions['Decision Tree'].append(dt_precision)
        model_recalls['Decision Tree'].append(dt_recall)
        model_f_scores['Decision Tree'].append(dt_f_score)
        model_training_times['Decision Tree'].append(dt_training_time)
        model_testing_times['Decision Tree'].append(dt_testing_time)

        # Print accuracy for current split
        print(f"Accuracy for Split {i+1}:")
        print("Neural Network:", nn_accuracy)
        print("SVM:", svm_accuracy)
        print("KNN:", knn_accuracy)
        print("Decision Tree:", dt_accuracy)
        print("--------------------")

    # Calculate average accuracy for each algorithm
    mean = lambda x: sum(x)/len(x)

    avg_accuracies = [mean(model_accuracies['Neural Network']), mean(model_accuracies['SVM']), mean(model_accuracies['KNN']), mean(model_accuracies['Decision Tree'])]


    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Algorithm': [algorithms[0]] * 10 + [algorithms[1]] * 10 + [algorithms[2]] * 10 + [algorithms[3]] * 10 ,
        'Accuracy': model_accuracies['Neural Network'] + model_accuracies['SVM'] + model_accuracies['KNN'] + model_accuracies['Decision Tree'],
        'Precision': model_precisions['Neural Network'] + model_precisions['SVM'] + model_precisions['KNN'] + model_precisions['Decision Tree'],
        'Recall': model_recalls['Neural Network'] + model_recalls['SVM'] + model_recalls['KNN'] + model_recalls['Decision Tree'],
        'F1 Score': model_f_scores['Neural Network'] + model_f_scores['SVM'] + model_f_scores['KNN'] + model_f_scores['Decision Tree'],
        'Training Time': model_training_times['Neural Network'] + model_training_times['SVM'] + model_training_times['KNN'] + model_training_times['Decision Tree'],
        'Testing Time': model_testing_times['Neural Network'] + model_testing_times['SVM'] + model_testing_times['KNN'] + model_testing_times['Decision Tree']
    })

    # Print the results table
    print("Results:")
    print(results_df)

    # Plot all metrics together
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time', 'Testing Time']
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        sns.barplot(x='Algorithm', y=metric, data=results_df, ax=ax)
        ax.set_title(f'{metric} Comparison')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

    # Calculate average accuracy for each algorithm
    avg_accuracies = results_df.groupby('Algorithm')['Accuracy'].mean()

    # Create a table for average accuracies
    avg_accuracies_table = pd.DataFrame({
        'Algorithm': avg_accuracies.index,
        'Average Accuracy': avg_accuracies.values
    })

    # Print the table with average accuracies
    print("Average Accuracies:")
    print(avg_accuracies_table)

    
'''
Decision tree
https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use
https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics

KNN

https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
https://www.analyticsvidhya.com/blog/2021/04/simple-understanding-and-implementation-of-knn-algorithm/

SVM
https://scikit-learn.org/stable/modules/svm.html
https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python

'''




    

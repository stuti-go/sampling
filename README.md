# Sampling Techniques and Model Evaluation

This project evaluates different sampling techniques for balancing a dataset and applies machine learning models to predict the target class.

## Sampling Techniques Used
1. **Sampling 1: Simple Random Sampling** - Randomly selects a fraction of the data.
2. **Sampling 2: Stratified Sampling** - Samples 80% of each class.
3. **Sampling 3: Systematic Sampling** - Selects every 5th sample.
4. **Sampling 4: Cluster Sampling** - Selects 50% of the data randomly.
5. **Sampling 5: Bootstrap Sampling** - Randomly samples the data with replacement.

**SMOTE (Synthetic Minority Oversampling Technique)** was applied as a balancing technique to generate synthetic samples for the minority class to balance the dataset before applying the sampling methods.

## Models Evaluated
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors
- Naive Bayes

The models were evaluated using **5-fold cross-validation** and their performance was measured using the **F1 Score**.

## Screenshot
![Screenshot 2025-01-18 192756](https://github.com/user-attachments/assets/2d0706b9-8e9f-4bdb-ba3e-561fa182c782)


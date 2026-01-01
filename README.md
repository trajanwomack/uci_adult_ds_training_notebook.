Readme includes notes on running the predictive model and outcome results 

How to run: 
- User must download the census data "Adult Data set" from https://archive.ics.uci.edu/dataset/2/adult
- User must then create own file path using the data or use existing written path
  example: train_path = 'data/adult.data'  

Porject write up: 
12/27/25

For this project I built that predicts whether an individual's income exceeds $50K per year based on census data features. The dataset includes both numeric features (age, hours worked per week) and categorical features (education level, occupation, marital status).
The key steps in this project included:
1.	A setup phase and problem definition phase
-	Initialization and install/importing of the necessary q pandas, NumPy, matplotlib, seaborn, scikit-learn, and optuna packages
2.	EDA (exploratory data analysis phase)
-	a data overview that included checking for missing values, creating general data distribution visualizations, and acquiring some basic insights. 
-	EDA provided essential context for building robust predictive models and highlighted the importance of handling both numeric and categorical features correctly.
3.	Data preprocessing pipeline (splitting x-feature and y-target) also creating numeric and categorical pipeline for x 
-	To prepare the data for modeling:

o	Separate features and target:
o	X contained feature predictor columns.
o	y contained the income target.
o	Feature type separation:
o	Numeric pipeline: Missing values imputed with the median and features standardized using StandardScaler.
o	Categorical pipeline: Missing values imputed with the most frequent category and one-hot encoded using OneHotEncoder.
o	Pipeline integration: Both numeric and categorical pipelines were combined using a ColumnTransformer to ensure consistent preprocessing for all features.
This approach ensured that preprocessing steps would be applied consistently across training and test data, preventing data leakage.

4.	Split splits of feature x and target 
-	The dataset was split into training (80%) and test (20%) sets:
•	X_train and y_train were used for model training.
•	X_test and y_test were reserved for evaluation.
•	Stratification was applied to preserve the proportion of income classes in both splits.
This step ensures that evaluation metrics reflect model generalization on unseen data.


4.5.	Application and Evaluation of Logistic Regression 

-	A logistic regression pipeline was applied to X_train:
•	The pipeline included preprocessing and the Logistic Regression classifier.
•	The model was fitted on X_train and y_train to learn feature coefficients.
•	Predictions were made on X_test to generate preliminary results.
This step provided a baseline model and allowed evaluation of a simple, interpretable approach to the classification problem.

-	Predictions from the logistic regression model were evaluated using:
o	Accuracy to measure correctness.
o	Precision, recall, and F1-score for class-specific performance.
o	Confusion matrix to visualize true vs. predicted classes.
o	Cross-validation was used on the training set to assess model and reduce variance in performance estimates.
This analysis helped identify the model’s strengths and limitations, guiding further improvement.

5.	We then apply a random forest predictive model and analyze the prediction models scores comparing them to the logistic regression

-	A Random Forest classifier was then applied to the same dataset:
•	Built as a pipeline including preprocessing steps and the Random Forest classifier.
•	The model captures non-linear relationships and interactions between features.
•	Predictions on the test set were compared to the logistic regression results.
Random Forest often outperforms simpler linear regression models in classification tasks, providing a stronger baseline

6.	Random forest predictive models can be improved by tuning hyper parameters, in this step we create hyperparameters that will be tuned and study a few iterations of the predictive model monitoring which hyperparameter stack is most ideal. 

-	Random Forest performance was further improved using hyperparameter optimization:
•	Defined ranges for key hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf).
•	Used Optuna to perform hyperparameter trials, evaluating each trial with cross-validation on the training set.
•	Monitored which combinations of hyperparameters resulted in the highest average accuracy.
This step is crucial for fine-tuning model performance while avoiding overfitting, and it identifies the most effective configuration for the Random Forest.

7.	The final step is applying the appropriate(best) hyper parameters to maximize the strength of our predictive model 

•	 The best hyperparameters identified by Optuna were applied to build a final Random Forest pipeline.
•	The model was trained on the full training set and evaluated on the test set.
•	Final evaluation metrics included:
o	Accuracy
o	Precision, recall, and F1-score
o	Confusion matrix
The optimized Random Forest model demonstrated improved performance over both the baseline logistic regression and default Random Forest, capturing complex patterns in the data while generalizing well to unseen examples.

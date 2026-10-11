# **Final Group Project Requirements** 

**Traditional Machine Learning Model and Deployable Application** 

|**Course**<br>__________________|**Section**<br>____________|
|---|---|
|**Instructor**<br>__________________|**Group**<br>____________|
|**Due date**<br>__________________|**Submissi**<br>**on**<br>____________|



## **Purpose and Expected Outcome** 

Each group must design, train, evaluate, and deploy an end to end traditional machine learning solution based on its approved project topic. The project must emphasize exploratory data analysis, compare exactly three appropriate algorithms under fair experimental conditions, and justify the final model using relevant evaluation metrics. The final output must be reproducible, clearly documented, and usable through a simple deployable application such as Streamlit. 

## **Project Scope** 

- The project must normally address a clear classification or regression problem with a defined target variable and intended user or decision context. 

- The dataset must be legitimate, ethically obtained, and appropriate for the stated problem. Public datasets and properly collected original datasets are allowed. 

- The primary technical emphasis is exploratory data analysis and evidence based model comparison, not interface complexity. 

- Unsupervised learning projects require prior written approval because their evaluation and deployment requirements differ from supervised prediction tasks. 

## **Permitted and Prohibited Methods** 

|**Category**|**Permitted Examples**|**Restrictions**|
|---|---|---|
|Classification|Logistic Regression, K Nearest Neighbors, Naive Bayes, Decision<br>Tree, Random Forest, Support Vector Machine, Linear or Quadratic<br>Discriminant Analysis, AdaBoost, Gradient Boosting|Select exactly three distinct algorithms<br>appropriate to the data and problem.|
|Regression|Linear Regression, Ridge or Lasso Regression, K Nearest<br>Neighbors Regressor, Decision Tree Regressor, Random Forest<br>Regressor, Support Vector Regression, Gradient Boosting<br>Regressor|Select exactly three distinct algorithms<br>appropriate to the data and problem.|
|Not allowed|Neural networks, deep learning, CNNs, RNNs, transformers, large<br>language models, pretrained foundation models, and AutoML<br>generated solutions|A prohibited method cannot be<br>counted as one of the three required<br>algorithms.|



**Important** Changing only the hyperparameters of the same algorithm does not count as a different algorithm. A simple baseline may be reported, but it does not replace any of the three required models. 

Final Group Project Requirements  |  Page 1 

## **Required Project Workflow** 

1. **Define the problem.** State the practical problem, prediction target, intended users, scope, and success criteria. Explain why machine learning is appropriate. 

2. **Acquire and document the data.** Identify the dataset source, license or permission, collection method, unit of analysis, number of records, features, target, and known limitations. 

3. **Perform exploratory data analysis.** Examine data quality, distributions, relationships, target balance, unusual values, and patterns that affect modeling. Every figure must lead to a written finding or decision. 

4. **Prepare the data.** Handle missing values, duplicates, outliers, encoding, scaling, feature selection, and class imbalance when applicable. Fit preprocessing steps only on training data to prevent leakage. 

5. **Train three algorithms.** Use the same data split, preprocessing logic, cross validation strategy, and primary metric for a fair comparison. Document the main hyperparameters and tuning procedure. 

6. **Evaluate and select the best model.** Compare validation results, confirm the selected model once on the untouched test set, and justify the choice using performance, interpretability, speed, and practical suitability. 

7. **Deploy the selected model.** Integrate the final preprocessing and prediction pipeline into a simple working application and test it using valid, invalid, and boundary inputs. 

## **Exploratory Data Analysis Requirements** 

EDA is a major component of the project. At minimum, the group must present the following evidence and explain its practical meaning: 

- Dataset dimensions, data types, feature definitions, units, target labels, and summary statistics. 

- Missing values, duplicates, inconsistent entries, impossible values, and the action taken for each issue. 

- Univariate distributions of important variables using suitable plots and numerical summaries. 

- Relationships between relevant features and the target, including correlation or association analysis when appropriate. 

- Class distribution for classification or target distribution for regression, including any imbalance concerns. 

- Outlier investigation that distinguishes data errors from legitimate extreme observations. 

- At least five meaningful and readable visualizations. Decorative charts that do not support a finding will not receive credit. 

- A concise EDA findings summary that directly informs preprocessing, feature selection, algorithm choice, or evaluation strategy. 

Final Group Project Requirements  |  Page 2 

## **Experimental Design and Model Comparison** 

Use an 80 percent training and 20 percent testing split unless the group provides a justified alternative. For classification, use stratification when possible. Perform model selection and tuning only within the training data, preferably using five fold cross validation. The test set must remain untouched until the final model is selected. Time ordered data must use a chronological split rather than random shuffling. 

|**Requirement**|**Classification**|**Regression**|**What to Report**|
|---|---|---|---|
|Primary metric|Choose and justify one such as macro<br>F1, weighted F1, recall, precision,<br>ROC AUC, or accuracy|Choose and justify one such as<br>MAE, RMSE, or R squared|State why the metric matches the<br>real problem.|
|Supporting<br>metrics|Accuracy, per class precision, recall,<br>F1 score, and confusion matrix; ROC<br>AUC when appropriate|MAE, RMSE, R squared, residual<br>analysis, and error distribution|Report the same metrics for all three<br>algorithms.|
|Validation|Same cross validation folds and<br>preprocessing pipeline|Same cross validation folds and<br>preprocessing pipeline|Report mean and variability across<br>folds.|
|Final test|Evaluate the selected model once on<br>unseen test data|Evaluate the selected model once<br>on unseen test data|Clearly separate validation and test<br>results.|



Accuracy alone is not sufficient for an imbalanced classification problem. The group must identify the cost of false positives and false negatives and select a primary metric that reflects that cost. 

## **Deployable Application Requirements** 

- Use Streamlit as the preferred framework. Another simple framework may be used only with instructor approval. 

- Load the saved best model and the same preprocessing pipeline used during training. 

- Accept practical user input through validated form fields or a properly formatted CSV upload. 

- Display the predicted class or value clearly. Show probabilities or confidence related outputs when the selected algorithm supports them. 

- Identify the model used and provide a short explanation of the input features, expected units, and output meaning. 

- Handle missing, invalid, and out of range inputs without crashing and display understandable error messages. 

- Include a deployed URL and complete local setup instructions. The application must run in a clean environment using the submitted requirements file. 

- Do not place passwords, API keys, private credentials, or computer specific absolute paths in the source code. 

Final Group Project Requirements  |  Page 3 

## **Required Deliverables** 

|**Numb**<br>**er**|**Deliverable**|**Required Contents**|**Accepted Format**|
|---|---|---|---|
|1|Deployable application|Working application using the selected best model, accessible through a<br>submitted URL and runnable locally|Streamlit URL or<br>approved equivalent|
|2|Source code|Notebooks or scripts for EDA, preprocessing, training, comparison,<br>evaluation, and application; saved model or pipeline; requirements file|Repository link and ZIP<br>archive|
|3|Technical<br>documentation|README, installation steps, folder structure, dependencies, run<br>instructions, input format, application guide, model limitations, and<br>troubleshooting notes|README and PDF or<br>DOCX|
|4|Dataset package|Original dataset, cleaned or processed dataset when created, data<br>dictionary, source link, license or permission, and processing notes|CSV or appropriate<br>open format|
|5|IMRaD style paper|Complete research style report with IEEE numbered citations and<br>references, model comparison, selected model justification, figures, tables,<br>and limitations|DOCX and PDF|
|6|Contribution record|Member names, assigned responsibilities, actual contributions, and<br>confirmation by all group members|Appendix in the paper|
|7|Ownership and<br>authorship declaration|Signed declaration identifying Dr. Armida P. Salazar and all student group<br>members as owners and authors of the submitted project output, subject to<br>university policy and properly acknowledged third party rights|Signed PDF; template<br>included in this<br>specification|



Final Group Project Requirements  |  Page 4 

## **IMRaD Style Paper Requirements** 

Use the IEEE conference paper format and IEEE numbered citation style. The recommended length is 6 to 10 pages, excluding appendices. Cite sources in the order they first appear using bracketed numbers such as [1] and provide a matching numbered reference list. All borrowed datasets, methods, definitions, code, figures, and claims must be properly acknowledged. 

|**Section**|**Minimum Content**|
|---|---|
|Title Abstract and Keywords|Specific title, concise problem, dataset, three algorithms, principal result, selected model, and<br>three to five keywords.|
|Introduction|Background, practical problem, research objective or question, significance, scope, and related<br>literature.|
|Methods|Dataset and ethics, features and target, EDA procedure, preprocessing, split strategy, three<br>algorithms, tuning, and evaluation metrics.|
|Results|EDA findings, comparison table for all models, validation results, final test results, confusion<br>matrix or residual plots, and application evidence.|
|Discussion|Interpret the results, explain why the selected model performed best, connect findings to the<br>problem, and discuss errors and tradeoffs.|
|Conclusion and<br>Recommendations|Answer the objective, summarize the selected model and practical value, state limitations, and<br>propose realistic improvements.|
|References|Complete IEEE numbered references with consistent in text citations.|
|Appendices|Additional figures or tables, data dictionary, sample inputs and outputs, group contribution<br>record, and signed ownership and authorship declaration.|



## **Required Submission Structure** 

```
GROUPNAME_PROJECTTITLE/
  app/                Streamlit application files
  data/original/      Original dataset and source note
  data/processed/     Cleaned dataset when applicable
  notebooks/          EDA and model development notebooks
  src/                Reusable preprocessing and modeling code
  models/             Saved best model or complete pipeline
  documentation/      Technical documentation and screenshots
  paper/              Final DOCX and PDF
  README.md           Setup and run instructions
  requirements.txt    Exact software dependencies
```

Final Group Project Requirements  |  Page 5 

## **Assessment Rubric** 

|**Criterion**|**Evidence of Quality**|**Points**|
|---|---|---|
|Problem and data definition|Clear target, users, scope, dataset legitimacy, provenance, and data dictionary|5|
|Exploratory data analysis|Complete data quality checks, meaningful visualizations, correct interpretation, and<br>findings that guide decisions|25|
|Preprocessing and<br>reproducibility|Appropriate transformations, leakage prevention, consistent pipeline, documented<br>random state, and runnable workflow|10|
|Three algorithm comparison|Three valid traditional algorithms, fair conditions, justified tuning, and complete<br>comparison|20|
|Evaluation and model<br>selection|Appropriate metrics, validation design, final test, error analysis, and defensible model<br>choice|15|
|Deployable application|Working and usable application, correct model integration, input validation,<br>understandable output, and successful deployment|15|
|Paper code and<br>documentation|Clear IMRaD style paper, correct IEEE citations, organized code, dataset package,<br>technical documentation, contribution record, and signed declaration|10|
|**Total**||**100**|



## **Minimum Acceptance Conditions** 

- All required deliverables must be submitted by the deadline and open successfully. 

- The project must compare exactly three permitted traditional machine learning algorithms. Failure to meet this requirement results in no credit for the algorithm comparison and evaluation criteria. 

- The deployed application and the submitted local version must use the same selected model and preprocessing pipeline reported in the paper. 

- Results must be reproducible from the submitted data and code. Fabricated, altered, or unverifiable results are unacceptable. 

- The ownership and authorship declaration must be complete and signed by the instructor and all group members. 

- Academic integrity and institutional policies apply. Undisclosed copied work, fabricated citations, or misrepresented contributions will be handled under applicable university rules. 

Final Group Project Requirements  |  Page 6 

## **Ownership and Authorship Declaration** 

This signed declaration is a required submission document. Complete all fields, obtain the signatures of the instructor and every group member, convert the signed copy to PDF, and include it in the documentation folder and as an appendix to the final paper. 

|**Project Information**|**Details**|
|---|---|
|Project title|____________________________________________________________|
|Course and section|____________________________________________________________|
|Group name or number|____________________________________________________________|
|Date submitted|____________________________________________________________|



### **Declaration Statement** 

We, Dr. Armida P. Salazar as the course instructor and the undersigned student group members, jointly declare that we are the owners and authors of the submitted project output identified above. The project output includes the application, source code, trained model and saved pipeline, exploratory data analysis, visualizations, technical documentation, research paper, and original materials produced specifically for this course project. 

We affirm that the submitted work accurately represents our documented contributions and that all external datasets, software libraries, code, models, frameworks, images, publications, and other third party materials have been properly cited, licensed, or acknowledged using IEEE citation format where applicable. This declaration does not transfer or claim ownership of pre-existing third party or institutional materials and remains subject to applicable National University policies, intellectual property rules, licenses, and Philippine law. 

### **Signatories** 

|**Name and Role**|**Signature**|**Date**|
|---|---|---|
|Dr. Armida P. Salazar<br>Course Instructor|||
|Student Group Member 1|||
|Student Group Member 2|||
|Student Group Member 3|||
|Student Group Member 4|||
|Student Group Member 5|||



Final Group Project Requirements  |  Page 7 

## **Final Submission Checklist** 

- ☐  Deployed application URL works 

- ☐  Repository link and ZIP archive are accessible 

- ☐  EDA notebook or script runs from start to finish 

- ☐  Exactly three traditional algorithms are compared 

- ☐  Best model is selected using justified evaluation metrics 

- ☐  Test set is used only for final evaluation 

- ☐  Original and processed datasets are included 

- ☐  Data dictionary and dataset source or permission are included 

- ☐  README and technical documentation are complete 

- ☐  IMRaD style paper is submitted in DOCX and PDF 

- ☐  IEEE in text citations match the numbered reference list 

- ☐  Group contribution record is signed or confirmed by all members 

- ☐  Ownership and authorship declaration is signed by the instructor and all group members 

**Submission note** The instructor may request an individual demonstration or explanation of any submitted code, analysis, model, result, or application feature to verify each member's understanding and contribution. 

Final Group Project Requirements  |  Page 8 


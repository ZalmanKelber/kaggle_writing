These series of notebooks explores the feature engineering, feature selection, model selection and construction of an ensemble of models for a submission to the [Linking Writing Process to Writing Quality](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality) Kaggle competition

A series of logs shows keyboard actions taken over the course of different SAT Writing essays.  Feature engineering begins by [constructing the individual essays from the keystroke information](https://github.com/ZalmanKelber/kaggle_writing/blob/main/construct_essays.ipynb).  Six separate dataframes with different kinds of features are then constructed from both the logs and the reconstructed essays.

[The first](https://github.com/ZalmanKelber/kaggle_writing/blob/main/writing6.ipynb?short_path=7aeb879#L661) relies on domain knowledge to extract features from the essay that are likely to represent either careful attention to detail or sloppy writing

[The second](https://github.com/ZalmanKelber/kaggle_writing/blob/main/writing6.ipynb?short_path=7aeb879#L661) uses a standard td-idf matrix from an n-gram of the reconstructed "words" in the essays (actual characters have all been replaces with 'q', so these are of limited use)

[The third](https://github.com/ZalmanKelber/kaggle_writing/blob/main/writing6.ipynb?short_path=7aeb879#L661) uses the timestamp data to analyze various properties about the lengths of inter-keystroke intervals (IKIs), including the length of various runs

[The fourth](https://github.com/ZalmanKelber/kaggle_writing/blob/main/writing6.ipynb?short_path=7aeb879#L757) uses aggregate data from the reconstructed essays to determine statistical data about the number and distribution of characters, words, sentences and paragraphs

[The fifth](https://github.com/ZalmanKelber/kaggle_writing/blob/main/writing6.ipynb?short_path=7aeb879#L910) analyzes count info for various types of keyboard actions

[The last](https://github.com/ZalmanKelber/kaggle_writing/blob/main/writing6.ipynb?short_path=7aeb879#L1164) analyzes changes in things like word count and cursor position per various numbers of successive keyboard actions

Scores for the essays range from 0.5 to 6.0.  Evaluating a set of score predictions that are all equal to the average score (around 3.7) yields a MSQE of close to around 1.0.  Evaluating predictions based on each of the six individual feature dataframes through an out of the box `GradientBoostingTree` model generates scores that range from around .85 to .65.  Using these individual GBT models to find the most useful features for each of the six dataframes without loss of accuracy, a composite feature dataframe is constructed with 174 features.  

Using model selection and grid search, the optimum model found is an ensemble that heavily weights an `XGBRegressor` model and combines it with (surprisingly) a simple `LinearRegression` model and a `GradientBoostingTree` model with specific hyperparameters, resulting in an MSQE of around .60
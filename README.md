<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br/>
<p align="center">
  <h3 align="center">Hotel Yearly Availability Prediction System</h3>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Questions</summary>
  <ol>
    <li><a href="#q1">How long did it take to solve the problem?</a></li>
    <li><a href="#q2">What software language and libraries did you use to solve the problem? Why did you choose these languages/libraries?</a></li>
    <li><a href="#q3">What steps did you take to prepare the data for the project? Was any cleaning necessary?</a></li>
    <li><a href="#q4">What machine learning method did you apply?</a>
    <li><a href="#q5">Why did you choose this method?</a>
    <li><a href="#q6">What other methods did you consider?</a>
    <li><a href="#q7">Describe how the machine learning algorithm that you chose works.</a></li>
    <li><a href="#q8">Was any encoding or transformation of features necessary? If so, what encoding/transformation did you use?</a></li>
    <li><a href="#q9">Which features had the greatest impact on salary? How did you identify these to be most significant? Which features had the least impact     on salary? How did you identify these?</a></li>
    <li><a href="#q10">How did you train your model? During training, what issues concerned you?</a></li>
    <li><a href="#q11">Please estimate the RMSE that your model will achieve on the test dataset.</a>
    <li><a href="#q12">How did you create this estimate?</a>
    <li><a href="#q13">What metrics, other than RMSE, would be useful for assessing the accuracy of salary estimates? Why?</a></li>
  </ol>
</details>

<!-- Q1 -->
<h3 id="q1">Q1. How long did it take you to solve the problem?</h3>

I spent 2 hours exploring the data and training a simple regression model with basic feature engineering techniques, such as normalization and encoding categorical features. After this first attempt, I spent 4 extra hours researching more advanced techniques across all learning processes and experimenting with a variety of combinations of feature engineering and models.

<!-- Q2 -->
<h3 id="q2">Q2. What software language and libraries did you use to solve the problem? Why did you choose these languages/libraries?</h3>

<h4>Language: Python</h4>
I used Python because it is the most common and powerful programming language in the data science domain with fancy and advanced libraries for solving a variety of problems.

<h4>Library: Pandas, Numpy, Matplot, Seaborn, Scikit-learn, Catboost</h4>
Pandas and Numpy are the basic libraries to load, manipulate, and save data in various formats with Python. Matplot and Seaborn are also the common libraries to visualize data and outcomes drawn from the calculation and manipulation for intuitive and effective communications. Scikit-learn is the most important library where most of the machine learning-related tasks are implemented from the development of machine learning models to the evaluation. Catboost is the open source library to use a machine learning algorithm that uses gradient boosting on decision trees. Catboost was one of the candidate algorithms that I chose to solve the given problem because of its high performance and efficiency.

<!-- Q3 -->
<h3 id="q3">Q3. What steps did you take to prepare the data for the project? Was any cleaning necessary?</h3>

Before diving into work on the machine learning model tasks, I performed the Exploratory Data Analysis (EDA) process as the very first step to look at how the data are structured and distributed and to take insights from the data for the subsequent process.

<h4>Overview of train and test data</h4>
I used the info() method to get fundamental information on the train and test data by skimming through it. There were one million rows in the data. The data was quite large. So, I was thinking that the sampling might be necessary for efficient analysis and modeling. But, I decided not to draw a sample because the amount of the rows was not bothersome in the process of analysis and modeling and the amount can be reduced by handling anomalies in the later step. Also, I recognized that there were four meaningful categorical features (jobType, degree, major, industry) except identifier features (jobId and companyId) and two useful numerical features (yearsExperience, milesFromMetropolis). At this point, none of the features had missing values according to the non-null counts.

<h4>Target - salary</h4>
I plotted a histogram to see how the salary was distributed and whether or not the distribution of the salary was skewed because just statistical figures were not enough to present the distribution for any type of audience or client with different backgrounds. The salary was drawn from the normal distribution(a.k.a. Gaussian distribution) as the distribution looked like a bell curve as seen below. Also, I found that there were a few rows whose salary was zero. These rows could be anomalies because only five rows had zero salary and the zero salary did make sense in the job post.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158277574-dc86e253-e69b-4e8f-a930-8c44750508d3.png" width="400" height="300">
</p>

<h4>Categorical feature - companyId</h4>
There were 63 unique companies. However, as this column name said, this feature was just an identifier for companies even though there were duplicate companies in that column. I was able to assume that this identifier feature would not be significant for the prediction, and I found that the impact of this feature on the prediction was insignificant based on the feature importance calculated with the catboost algorithm in the later step.

<h4>Categorical feature - jobType</h4>
According to the count plot, the frequencies of each category were very similar. Based on the box plot and line plot, as I expected, the higher a position was, the higher its salary was. The box plot is a good method to examine the distribution of the data in statistical perspectives and to detect anomalies out of the distribution range. The line plot is used to look over how continuous values are changed depending on categories. Of course, CEO is the position with the highest mean salary. According to the box plot and statistical measurements by describe() method, CTO and CFO positions had the same mean salary as these two positions are generally at the same level position but with different specialties. I thought that I needed to encode this jobType feature using the ordinal encoding technique, and it might be good to merge CTO and CFO categories into one category.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158277786-e45f46e3-7219-4687-aa4f-cafbdf3d7bcc.png" width="600" height="500">
</p>

<h4>Categorical feature - degree</h4>
The frequencies of each category in the degree feature were different depending on the college admission. The frequencies between college degrees were very similar. However, the frequencies of HIGH_SCHOOL and NONE were more than the college degrees’. As I expected along with the job position, the higher a degree was, the higher its salary was. The doctoral degree presented the highest mean salary. The mean salary was significantly different depending on the college admission showing the steep slope between BACHELORS and HIGH_SCHOOL on the line plot. At the beginning, I thought that NONE category could be merged into the HIGH_SCHOOL category because these two categories showed the similar trend in the box plot. But, after I tested a prediction model with a new merged category with HIGH_SCHOOL and NONE, I realized that the performance after this merging process did not outperform the performance without the merging task. Also, in the line plot, the mean salary of HIGH_SCHOOL was a little larger than the one of NONE. Thus, I concluded to leave them as they were. I also needed to encode this feature using the ordinal encoding technique.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158278131-a733592e-1c97-4e1b-bb95-ae3e19c8d02c.png" width="600" height="500">
</p>

<h4>Categorical feature - major</h4>
The frequencies of each category in the major feature were not significantly different when NONE was not considered. NONE mostly came from the job posts whose degree was HIGH_SCHOOL or NONE because the major is associated only with the college degrees. However, I found that within job posts whose major was NONE, some of the posts whose degree was one of the college degrees existed. I was sure that college degrees without majors did not make sense, and then I tried to assign one of the majors which was the most frequent among a group that had the same conditions in the jobType, major, and industry columns. But, this imputation did not make improvement in terms of the performance. Thus, I decided not to impute NONE of the job posts whose degree was one of the college degrees. The mean salary of NONE in the major feature also showed the significant difference like the degree case with the steep slope in the line plot. The interesting thing was that ENGINEERING and BUSINESS majors presented the higher mean salary as we have known this truth. For the major feature, I conducted the one-hot encoding because the major is not an ordinal category.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158278195-5b55ee31-5010-4522-b24f-59795d46b49c.png" width="600" height="500">
</p>

<h4>Categorical feature - industry</h4>
The frequencies of each category in the industry feature were similar to each other. I discovered that OIL and FINANCE industries that have the largest capital flows traditionally presented the higher mean salary. On the other hand, the education industry showed the least amount of the mean salary.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158278230-0486d2fe-8e68-4810-ae7f-223fdfc69fac.png" width="600" height="500">
</p>

<h4>Numerical feature - yearsExperience</h4>
As I expected, the higher work experience years were,  the higher the mean salary was. Therefore, the yearsExperience feature had a positive relationship with the salary.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158278268-741b4547-d223-4ca0-9ab3-d22562aa1e48.png" width="400" height="300">
</p>

<h4>Numerical feature - milesFromMetropolis</h4>
I found that jobs whose distance from the metropolitan areas were far had lower mean salaries. Thus, the milesFromMetropolis feature had a negative relationship with the salary.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158278284-089c17ca-3548-4a39-b305-a919130b5871.png" width="400" height="300">
</p>

<h4>Correlation</h4>
By using heatmap for correlation, I was able to confirm how numerical features were associated with the salary.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158278294-9feacd5e-6f20-48ca-a8dc-c928ed8353c4.png" width="400" height="300">
</p>

<!-- Q4 -->
<h3 id="q4">Q4. What machine learning method did you apply?</h3>
Machine learning method: CatBoostRegressor

<!-- Q5 -->
<h3 id="q5">Q5. Why did you choose this method?</h3>
CatBoost is an algorithm based on the gradient boosting which builds models sequentially following the ensemble learning process by trying to reduce the errors of the previous weak learners based on the gradient descent optimization until a better model comes out with the better performance. In general, the gradient boosting algorithm for regression shows better accuracy than other regression techniques, such as linear regression, which leads many researchers and engineers to using it in competitions. The gradient boosting has many advantages over the other algorithms in that it does require to spend less time preprocessing the data including handling missing data. Thus, this algorithm is a good fit to solve the problem with categorical and numerical features. The higher flexibility to tune hyperparameters and use diverse loss functions is also a benefit. There are some other machine learning models based on this algorithm, LightGBM and XGBoost. All these three models are great and similar in terms of time consuming and accuracy. But, the CatBoost is very convenient and efficient to use categorical features as its name was coined from category and boosting. Our data has four categorical features and two numerical features that I have to focus on. This is the reason why I chose the CatBoost.

<!-- Q6 -->
<h3 id="q6">Q6. What other methods did you consider?</h3>
I considered the LightGBM and XGBoost models, as I described that these techniques are also powerful above, and I tested them too. The XGBoost generally provided high accuracy but it took longer than the other models to train a model. The LightGBM showed the great performance and learning speed as the CatBoost did. In the end, the CatBoost was better than the LightGBM for the accuracy and the convenient categorical feature engineering on the salary data.

<!-- Q7 -->
<h3 id="q7">Q7. Describe how the machine learning algorithm that you chose works.</h3>
The CatBoost algorithm is based on the gradient boosting with decision trees. The algorithm comprises a number of decision trees, each of them is added sequentially from a single initial tree with better performance in terms of loss compared to the previous trees. To make a single tree, the algorithm splits the numerical features into buckets for the threshold in each leaf. For the categorical features, the algorithm encodes these features into numerical features by statistically calculating a variety of combinations among categorical features or between categorical and numerical features based on the label value. After generating pairs of the splits in the previous step, some of the pairs that present lower penalty from various penalty functions are selected to put them into leaves to finalize the structure of a tree. With this tree structure, the algorithm creates the ensemble model with a number of the selected trees.

<!-- Q8 -->
<h3 id="q8">Q8. Was any encoding or transformation of features necessary? If so, what encoding/transformation did you use?</h3>
For the given regression problem, it was necessary to encode categorical features, jobType, degree, major, and industry, into numerical ones. I noticed that jobType and degree features are ordinal categories and the higher level of the occupation or the education attainment were likely to get more salaries than the lower levels. Thus, I thought that the ordinary encoding method was required for these ordinal categories and one-hot encoding method for the major and industry features. However, I tested which is better between encoding the categorical features by myself and passing them to be transformed automatically in CatBoost’s built-in method. There was no difference so I chose the latter method.

<!-- Q9 -->
<h3 id="q9">Q9. Which features had the greatest impact on salary? How did you identify these to be most significant? Which features had the least impact on salary? How did you identify these?</h3>
The jobType feature had the greatest impact on the salary. The companyId feature had the least impact on the salary before I excluded it from the train data. After the companyId was excluded, the major became the least important feature. The yearsExperience feature also presented a high impact on the salary. To get this feature importance result, I used SHAP (SHapley Additive exPlanations) instead of the built-in method, get_feature_importance(), because SHAP estimate provides better accuracy and more detailed information even though its complexity is higher than the traditional method. SHAP value is the measurement to estimate how much each feature value in all features contributes to the prediction of each instance.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158281587-6a8b4854-b6de-4a72-a097-7c3022f6353b.png" width="500" height="300">
</p>

I also used summary_plot() in SHAP to get detailed impact or relationship with a salary by encoding the categorical features separately. For example, according to the figure below, JANITOR in the jobType feature had the strong impact on the lower salary.
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158278710-d4297276-6479-4654-afa4-5a5e13821632.png" width="600" height="600">
</p>

<!-- Q10 -->
<h3 id="q10">Q10. How did you train your model? During training, what issues concerned you?</h3>
After exploring the features and their impacts on the salary, I performed the feature engineering and the hyperparameter tuning as follows.

<h4>Imputation</h4>
I assumed NONE values in the degree and major features as the missing values. The mean salary between HIGH_SCHOOL and NONE was a little different. It did not make sense that the job posts whose degree was one of the college degrees had NONE values in the major feature. 

<h4>degree</h4>
I replaced NONE values with HIGH_SCHOOL to make these two values to belong to a single value as the non-college degree.

<h4>major</h4>
I replaced NONE values in the major but having one of the college degrees with the most frequent major within a group that presented the same jobType, degree, and industry. For example, there is a row whose jobType is CEO, degree is MASTERS, major is NONE, and industry is OIL. I created a group that consists of rows whose jobType is CEO, degree is MASTERS, and industry is OIL. If ENGINEERING is the most frequent major in this group, I replace NONE with ENGINEERING. I also tested adding one more condition on this methodology which was the salary range between maximum and minimum in the box plot.

In conclusion, these ideas did not improve the performance of the model. Rather, the performance got worse when I applied the imputation to both degree and major features. Therefore, I decided to leave NONE values as they were.

<h4>Handling anomalies</h4>
I chose to drop anomalies. First, I removed the job posts whose salary was zero. This worked itself to reduce the RMSE estimate. To deal with outliers above the maximum of the box plot in each category across all categorical features, I picked the highest upper whiskers among the categories in each categorical feature and put them into a list. Then, I picked the highest one again in the list, and I removed all job posts whose salary was above this most highest threshold salary. I tried to pick a threshold salary based on the mean value instead of the max value. But, it caused the loss of valuable data points and it did not make an improvement on the performance.

<h4>Normalization</h4>
I used StandardScaler() in the scikit-learn library to normalize scalar values of the yearsExperience and milesFromMetropolis. However, this process did not make any change in improving the performance of the model because the variances of these features were not too large enough to do normalization.

<h4>Drop columns</h4>
I dropped the identifier columns which were jobId and companyID.

<h4>Encoding categorical features</h4>
I encoded the two ordinal features, jobType and degree, using OrdinalEncoder in the scikit-learn library, and the two nominal features, major and industry, using the get_dummies() method for the one-hot encoding. As I trained the CatBoost model that has its own built-in encoding method, I compared the performance when I encoded these features in the preprocess step and when I passed the features into the training process without the data preprocessing. I found it was better to leave them up to the CatBoost algorithm.

<h4>Tuning hyperparameters</h4>
I utilized the Bayesian optimization technique to tune the hyperparameters of the CatBoostRegressor. The Bayesian optimization method has an advantage over the grid search and random search methods in that it takes less time with a competitive set of hyperparameters than the grid and random search. Also, unlike the grid and random search methods that independently experiment a variety of combinations of hyperparameters, the bayesian optimization method searches the next hyperparameter candidates based on the result from the previous iteration. This algorithm works very similarly to the gradient boosting algorithm that I used to solve the given salary prediction problem here.

<!-- Q11 -->
<h3 id="q11">Q11.Please estimate the RMSE that your model will achieve on the test dataset.</h3>
<h4>Baseline model</h4>
RMSE:18.81 / MAE 15.29 / R2: 0.76 (BEFORE FEATURE ENGINEER)
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158279015-567ecddc-acd2-4517-9e70-63d61562b16d.png" width="400" height="300">
</p>

RMSE:18.65 / MAE 15.18 / R2: 0.76 (AFTER FEATURE ENGINEER)
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158279028-3faf62b4-bc37-4ab5-b782-cf2dc314ad00.png" width="400" height="300">
</p>

<h4>Optimized model</h4>
RMSE:18.65 / MAE 15.18 / R2: 0.76 (AFTER HYPERPARAMETER TUNING)
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/42654506/158279046-f2d6e72d-769f-4c46-9608-38fec8cdec3e.png" width="400" height="300">
</p>

<!-- Q12 -->
<h3 id="q12">Q12. How did you create this estimate?</h3>
When I preprocessed the train data and I trained a baseline model without tuning hyperparameters, I got 18.65 that was less than a baseline model without both preprocessing the data and tuning hyperparameters. However, the estimate was not improved even after hyperparameters were tuned by the Bayesian optimization. 

<!-- Q13 -->
<h3 id="q13">Q13. What metrics, other than RMSE, would be useful for assessing the accuracy of salary estimates? Why?</h3>
In my humble opinion, RMSE is still a reasonable metric for the given problem to predict salaries. The right choice for a metric depends on a circumstance and where I put more focus on. In the given problem, job seekers generally find jobs that can be more beneficial in terms of salary if other requirements are met. When a job provides more salary than its predicted one, then a job seeker would not be disappointed with this error. On the other hand, if an actual salary is less than a predicted one, it is more likely that the job seeker would be unsatisfied with the erroneous outcomes. Thus, RMSE would be good because the metric penalizes the over-estimates. If I need to find a different metric, I think that MAE is also a useful metric. MAE does not penalize bigger errors so it is easier and more intuitive to interpret the model performance and it is not sensitive to outliers. In the given problem, it would be hard to say that salaries out of the specific range determined by the mean salary are outliers because determinants of salary are very complicated and there are lots of considerations to estimate a salary. Even though every condition is equal in position, major, degree, and industry, a salary can be very diverse depending on company scale, country, and so on.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jooyeongkang/
[product-screenshot]: images/screenshot.png

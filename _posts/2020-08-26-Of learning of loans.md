---
layout: post
title: Creating a Predictive Model
subtitle: Misadventures in Learning

---
## How it all started
[TP_Shap](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20Shap%20TP.jpg)
[TP_Row](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20Shap%20True%2B.jpg)
[RF_Models](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20RF.png)
[LR_Models](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20LR.png)
[XGB_Models](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20XGB.png)
[LR_Confusion Matrix](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20LR%20SMOTE%20Confusion%20Matrix.png)

Remember color-by-number from childhood? Some kids hated those. Me, I had my mother make regular pictures into color-by-number, now I knew exactly what to do! Fast forward a few years. “Pick a topic for your paper…”. That’s the hardest part, then to make it long enough... I would rather prove a theorem, thank you.

Fast forward many more years; I have forgotten almost everything about proving theorems. Hence, I am studying Data Science at Lambda School. It’s time to begin my Unit 2 Build Project. “Pick a data set…” Ugh, the worst part. I found this data set on Kaggle, it seemed tidy and was brand new. I was so busy learning the mechanics from my lessons, I didn’t notice a problem. Little did I know, a highly imbalanced data isn’t a great choice for a beginner. Trying to find a way to distinguish a tiny amount of data from all of the rest proved to be tricky. I learned about a new technique. Soon everything was going swimmingly. Then, the weekend before the project was due I happened upon a bit of information that set off that little niggling feeling… Sure enough, I had misunderstood how to apply the new technique. What I thought was working well was hardly working at all… I was back to looking for needles in a pin stack, everything looks alike.

## The Project
### The Data

TVS is a lending institution which provides both secured and unsecured loans. A secured loan has something which can be repossessed if a customer defaults, for example, a two-wheeled vehicle. A personal loan is an example of an unsecured loan, if the customer fails to pay, the money is lost.

The data set is comprised of information on TVS customers who have already taken out a loan for a two-wheeled vehicle and would like to take out a personal loan as well. Around 2% of TVS borrowers default on their vehicle loans. If TVS can predict who will default on one loan, they will know not to approve another.

There is no data available about the people whose loan applications were denied. Hence, defaulters cannot be compared to the denied clients to find similarities. All of the clients have already passed through the TVS loan approval process.

Initially there were about 120,000 clients listed in my data set, with 32 possible pieces of non-identifying information about each one. This client information is referred to as features or columns. The TVS data was missing a lot of information, both clients and columns with too many holes to be dropped. Several other columns with information gathered after the loan was granted had to be dropped as well. Now there were about 85,000 clients and 11 columns of information.

Now it was time to begin creating a model to detect defaulters. The baselines for all of my basic models were abismal at 0. None of the models identified a single defaulter!

### Balancing the Data
I tried two techniques for balancing the data. Downsampling was done by cutting down list of the non-defaulters to approximately the same number as defaulters. The resulting models found 79-89% of the defaulters! Unfortunately, they also labeled 74-89% of the other clients defaulters. Better, but still far from useful.

SMOTE, Synthetic Minority Oversampling Technique, SMOTE works the opposite of downsampling. SMOTE was used to create a fictional population of defaulters to balance out the non-defaulters. It sounds a little fanciful, but is proven to be legitimate in the world of data science. This technique resulted in models slightly better than the models made with downsampled data.

I applied an ROC curve to my models to get my best results yet. The Random Forest model had a baseline of 52 on validation data, while the baseline for the Logistic Regression model was 67.

![Val_Model](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20Best%20Validation%20Models.jpg)



In the end the best model I created was a Logistic Regression model using SMOTE data and an ROC curve. The baseline for this model was 54.
![Best_Model](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20Best%20Model.jpg)

Here's an easier way to look at the information for my best model.

![Best_CM](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Best%20Model%20Confusion%20Matrix.jpg)

Unless TVS wants to sell a large percentage of their personal loans to another lender, my models are not particularly useful. However, school projects are intended to be learning experiences, so the most important goal was achieved.

### Let's Pretend
TVS isn't likely to find my models useful, however let's pretend for a moment that the model described below is actually fantastic. TVS adopts my model and a client wants to know why he was denied a loan. "We plugged your information into our nifty formula and it said not to give you a loan." Isn't going to work. However, there is a helpful way to display the how the model selected it's results. This Shapley, not shapely, plot will help the banker descibe the reason behind the denial.
![Borrower](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20Shapley%20Row.png)
Above is the information about a particular client. Below is the Shapley plot identifying how those pieces of information contribute to whether or not his loan will be approved. He is 40 years old, owns a scooter and has only one secured loan and appears to live in a preferred neighborhood. All of these attributes suggest that he is a good client. However, this does not outweigh the fact that he is self-employed, has no history of unsecured loans and chose not to make advance payments on his current vehicle.
![Shapley_TP](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20Shapley%20TP.png)

"Sir, you are self-employed with no history of taking out unsecured loans, in addition to all of this other information we didn't share with our data scientist we have decided that it is not advisable for us to extend additional credit at this time."



#### Other useful information

![All_Models](https://raw.githubusercontent.com/SaraWestWA/SaraWestWA.github.io/master/assets/img/Unit%202%20All%20Models.jpg)

Find my raw data here: [TVS](https://raw.githubusercontent.com/SaraWestWA/DS-Unit-2-Applied-Modeling/master/TVS.csv)

Find the data on Kaggle: [Kaggle_TVS](https://www.kaggle.com/sjleshrac/tvs-loan-default)

Take a peek at my code here: [Build_Notebook](https://raw.githubusercontent.com/SaraWestWA/DS-Unit-2-Linear-Models/master/module4-model-interpretation/SW_DSTP7_Unit_2_Build.ipynb)

##### Eat honey for it is good; honey from the comb is sweet to taste.
Proverbs 24:13 paraphrase








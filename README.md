![](http://rentinginla.com/wp-content/uploads/2015/12/Buying.jpg)
# Predicting House Prices with Advanced Regression Techniques

The goal of this project is to predict home prices given the features of the home, and interpret the models to find out what features add value to a home! The dataset for this project was taken from Kaggle Ames housing price Prediction competition https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Buying a house for any average person is a major decision in his/her life and a lot of factors contribute to the pricetag for a dream house.  Intuitively even before performing machine learning, one could predict that neighborhood (being close to school, highway, shopping centers), kitchen quality and overall quality of house are major influence.

This dataset contains detailed information about houses in Ames, Iowa from 2006 to 2012, covering square footage, neighborhood, number of bedrooms, bathrooms and exterior materials. Preliminary examination of the dataset shows there are heterogeneous features with a mix of ordinal, continuous and nominal attributes. 
* Continuous features contain area information such as size of lot, bedroom, kitchen and basement. 
* Nominal features describe types of exterior and garage material, neighborhood and sale type while 
* Ordinal features describe quality and condition of house, garage, basement and material. 
* In addition many of the features contain null values which have to be handles before feeding the dataset to machine learning models. 

## Reading Data
I read the data with pd.read_csv method with 'keep_default_na' = False parameter, which does not automatically convert "NA" values to np.nan. I employed this strategy to separate the mislabelled data from missing data. Many of the features have mislabelled data where "NA" simply meant "no_feature" instead of data not available. These mislabelled data have to be changed manually into correct labels and should not be considered during data imputation. There are 12 such mislabelled features.

## Data exploration, data imputation and feature engineering:

I set out to perform in depth EDA due to the above mentioned charater of this dataset. A good analyst will always understand data well before fitting predictive models, because feeding dirty data into a model will never give a satisfactory result. 

1) Any feature that was more than 95% empty, I discarded it as there is simply not enough information available to fill in missing values. These features probabaly have no predictive power. I deleted 7 such features
2) 3 features had wrong datatype. I felt 'MoSold' and ' YrSold' should be string type as numerically they can not be used in meaningful calculations.
3) For features with numeric data types, I imputated missing values with median as data is highly skewed (we will see that through distribution) and for categorial type features I used the most common value to impute missing values.
4) I converted ordinal features into numeric labels as their order have values have some inherent order. 
5) I combined very closely related feature into one super feature, combining (Basement quality, basement condition), (overall quality, overall condition) and (open porch, screen porch and enclosed porch area)

 
## Examining prediction values: SalePrice:
it is important to examine the predictions by understanding its distribution and what pattern does it follow. SalePrice is righ skewed, with majority of houses around $150,000 pricetage whereas very small number of houses are above $450,000. Extreme expensive houses can be considered outliers and removed as keeping them inside model will pull our prediction towards higher end and reduce accuracy. I am going to log transform SalePrice to remove skew before removing the outliers. I followed Tukey's rule to determine outliers and found 5 such datapoints.

#### Understanding distribution of features and their relationship with SalePrice
I created heatmap of correlation matrix using seaborn, python visaulization package. The heatmap clearly indicates that few features have really high correlation, upto 0.8. I decided to keep only one of these features as I want to minimize redundancy in data.

Upon creating scatterplot of numerical features with SalePrice, features like, OverallQual, KitchenQual, GarageArea, LotFrontage and GrLivArea has high high correlation. I expect some of these features to show up in my model. 

Distribution of numeric feature showed lot of right skew. Many machine learning linear model require input data to be normally distributed, hence I log transformed numeric feature to remove skew and reduce outliers.

## Linear models for building predictions:
As it is regression problem, I started out with Linear REgression to fit the datapoints. I used 2 metrices, R2 score and mean_squared_error to understang teh performance of linear model.

As expected regularised models Lasso (with L1 penalty score) and ElasticNet (with both L1 and L2 penalty) were superior to simple linear regression. Regularization tends to make coefficients of certain features as zero and reducing their predictive power. As we already know that this dataset had lot of collinear features, regularization is definitely our friend.

Absolute measure of coefficients given by these models shed light on features that are really important. 

I also used DecisionTreeRegressor which provide feature importances. Finally GrLivArea, OverallQual and Neighborhood were picked up by my models to be most predictive of the SalePrice




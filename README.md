# Pattern-Recognition

## Objective 
 
Given a user representation based on the attributes described above, predict the rating of the 
Hotel.  We formulate this as a Multi-class classification problem wherein user-ratings are 
categorized into 5 classes 
While in real life, there is a simple linear correlation between hotel prices / location with their 
ratings. Pricey Luxury Hotels tend to have better ratings than cheap motels and thus it does not 
make for an interesting machine learning problem.  

## Section 1 Introduction
### Dataset(s) 
● For this project, we choose the public dataset scraped from Booking.com. It contains 515k customer reviews and scoring of 1493 luxury hotels across Europe.  
● Source - Kaggle 
  ○ This dataset is hosted on Kaggle platform: 
  ○ https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe 
● Size -  277MB 

## Section 2: System design/architecture/data  
The architecture of the system shown below describes the overall process. The major steps involved in predicting a hotel’s rating is as follows: 
● Data Cleaning & Pre-Processing 
● Train the Model 
● Predict the Rating using the model

### 2.1 Data Cleaning & Pre-processing technique:  
The first step in our preprocessing is handling the highly imbalanced data. After analysing the data the reviewer score was in floating points, the first step was the convert them to integer 
values to understand the number of ratings better.
To classify them into different classes, it was important to understand the number of ratings each class has, and then convert them into bins.  The analysis leads to classification of ratings into 5 classes: 
● Ratings from 0 - 5 : Class 1 
● Ratings from 6 - 7 : Class 2 
● Ratings from 8     : Class 3 
● Ratings from 9     : Class 4 
● Ratings from 10    : Class 5 

Since the data consisted of text, we start with text normalization. Text normalization includes: 
● Converting all letters to lower case - it helps in mapping same words in different cases to map to same word.  ● Removing numbers, punctuations, accent marks and other diacritics 
● Removing white spaces 
● Removing stop words, sparse terms, and particular words - Stop words are a set of commonly used words in a language. Examples of stop words in English are “a”, “the”, “is”, “are” and etc. The intuition behind using stop words is that, by removing low information words from text, we can focus on the important words instead. 
● Removing words shorter than 4 characters 
● Stemming, lemmatizing : 
  ○ Stemming is useful for dealing with sparsity issues as well as standardizing 
  vocabulary. 
  ○ Lemmatization on the surface is very similar to stemming, where the goal is to 
  remove inflections and map a word to its root form. 
  
### 2.2 Technologies & Tools used (and why) 
● For our project we required lots of scientific calculations so we used NumPy and Pandas, which provides a lot of convenient and optimized implementations of essential mathematical operations on vectors. 
● To remove stop words of English dictionary from the Positive review and Negative reviews column we used ntlk’s stopword module as a preprocessing step. 
● We used geopy and opencage geocoder for getting Country, City and Postal Code of hotels of our data. 
● We used TF-IDF, for vectorization of train, test data and validate data. TF-IDF enables effective clustering process. It normalizes word frequency in terms of their relative frequencies present in the data. TF-IDF reduces the importance of common terms in the data and ensures that the matching of documents be more influenced by that of more discriminative words which have relatively low frequencies in the collection. ● We converted our data into a csr_matrix as it provides a faster matrix vector product and  it is efficient for row slicing. Also our matrix is mainly composed by zero elements, so we can save space memorising just the non-zero elements. R 

## Section 3 Experiments 
### 3.1 Methodology followed 
● We did research on getting the best ratio for splitting the data. We found few comments and followed that. Below are few comments. 
  ○  The best result was obtained when 20% of the data were used for validation and the remaining data were divided into 70% for training and 30% for testing. 
  ○  For a small data set, it is good to use k-cross validation and  70-15-15 ratio. For big sets it is fine to use circa 75-25 or more probs. 
  ○ Most of the researchers follow 60 for training and 20,20 for validation and testing. 
  ○ If it is large enough, 66% split is a good choice 
  ○ If you have a large data set, it's ok to consider the training:testing ratio 75:25. So we split the data as below, n-fold -cross validation: No of folds - 3  size of training - 66%  test set - 17% validation set - 17% 

### 3.2 Modelling:  
The problem we will solve is a binary classification task with the goal of predicting the hotel rating by an user.After we are done with the pre processing of the data we have trained our models with the below mentioned algorithms. 
● Random Forest Classifier 
● Neural Networks MLPClassifier 
● Naive Bayes Classifier 
● Support Vector Machine 

## 4 Conclusion 
Based on the results, we concluded that Random Forest is a good machine learning algorithm to use for predicting star ratings based on user reviews. It generates an F1 score of 0.91 which is best so far. It can also be understood that the dataset being chosen should have
reviews distributed across all star ratings. This will help the model to train better. For example, if a dataset is chosen in such a way that a big majority of the user reviews are 5 stars, then the model is likely to misclassify most reviews as 5 star ratings. This will bring down the performance. 

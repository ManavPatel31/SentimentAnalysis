# Sentiment Analysis using NLP
Sentiment Analysis Using Natural Language Processing on Amazon's All_Beauty Dataset.


Sentiment Analysis and Recommender Systems 
 All Beauty Dataset

 
# I.	INTRODUCTION 
The dataset is a collection of data on beauty products sold on Amazon. It contains 371,345 reviews across 32,992 products and the overall ratings are ranged from 1 to 5. Each product in the dataset is described by a set of features such as ‘Style’ which provides a detailed description of the product and also ‘reviewText’ which contains a detailed review of the product given by the customer. This dataset can be used to research to try and understand customer reviews and help achieve better Sentiment understanding to the model. In this report we will look at the sentiment analysis done using NLP to make a recommender system to recommend products based on customer’s personal preference.

# II.	PROJECT SCOPE

# A.	Scope
The All_Beauty dataset can be used to perform sentiment analysis and build a recommender system. Sentiment analysis involves examining customer reviews to determine whether they have a positive, negative, and neutral sentiment towards a product. A recommender system involves providing suggestions to customers based on their preferences and behaviours. 


# B.	Problem
Many different factors can influence a customer’s opinion of a product which defines the main problem with building a sentiment analysis model and a recommender system. For example, a single customer an have different opinions on different beauty products provided by the same company, while other customer do compare prices and brands. 

# C.	Proposed Solution
To address these challenges, we will build models that will understand the different factors that influence customer behaviour. For example, sentiment analysis models can be trained to take into consider different emojis and punctuations and specific keywords to try and understand different human sentiments such as excited, terrible, sarcasm, etc. The recommender systems can be trained to identify patterns based on customer behaviours such as purchasing history and product preferences.

# III.	DATA EXPLORATION
The original dataset contains around 371,345 reviews for Beauty products on amazon out of which we selected a part of this dataset containing 5,269 reviews across 85 distinct products. After loading the JSON file, we explored the distribution of the data across different categories such as ‘asin’,’overall’ and ’reviewerID’, and found insights to the correlation amongst which products have the most reviews and how those reviews are distributed among different star ratings. We also calculated the total number of reviews given by a single customer. 
We loaded, explored and merged the meta_All_Beauty.json dataset to get the product name and found the first five most reviewed product and the most popular product is “Bath &amp; Body Works Ile De Tahiti Moana Coconut Vanille Moana Body Wash with Tamanoi 8.5 oz”. We found the top five products and displayed the counts of reviews. 
<img width="252" alt="image" src="https://user-images.githubusercontent.com/90365773/233864880-cb134fc0-d539-46f1-835d-3a66c8cbdb4b.png">
Fig.1<br/>

Fig.1 shows a Time Series Analysis which provide more valuable insights of the most popular product across a certain period.

# IV.	PREPROCESSING AND TEXT REPRESENTATION
To perform preprocessing and text representation, we randomly selected a sample of 830 reviews and created a new data frame. A function was applied to label the ratings of the products as positive, neutral, and negative based on the overall ratings of the product, the result of which was then stored in a new column.

To continue performing this phase we dropped the unwanted columns which were not applicable to the text representation. We loaded two text files with positive and negative words which would help us enhance the results. All the reviews were lower cased and stored in a new column in the data frame.

We imported the stop words corpus from NLTK and applied the word_tokenize() to the review_processed_docs to split each document and filter out the stop words from each document. We created a Tfidf representation using Scikit-learn’s TfidfVectorizer. We printed the idf values for all the words and the Tfidf representation for all words in the corpus. We have used Tfidf because it addresses the weight frequency of each term by using its overall frequency across the entire document, and by calculating the inverse document frequency it measures how rare the term is across the entire corpus. It does not consider the irrelevant words that affect the determining of sentiment. It also gives us the sparse matrix representation which helps to reduce the risk of overfitting in the model. We used Word2Vec because it can handle out of vocabulary words by giving them meaning based on the context in which they appear. It is suitable for large scale sentiment analysis tasks.



# V.	SENTIMENT ANALYSIS USING LEXICON APPROACH

# A.	Using Vader

We chose to use the Vader sentiment analysis model because it can handle sentences with typical negations, use of conventional punctuation, understanding slang, understanding emoticons, emojis. Stopword’s removal does not give better result in Vader because it uses word such as ‘but’ in calculating sentiment. 

“VADER is VERY SMART, handsome, and FUNNY!!!”
{'pos': 0.767, 'compound': 0.9342, 'neu': 0.233, 'neg': 0.0} 

Catch utf-8 emoji such as 💘 and 💋 and 😁
{'pos': 0.279, 'compound': 0.7003, 'neu': 0.721, 'neg': 0.0}

Here are the two examples that show how sentiments in these two different sentences are determined. The first example shows us the acceptance of punctuation by the model in analyzing sentiments and the other example shows the acceptance of emojis in determining sentiments. The parameters ‘pos’ shows the overall sentiments (mainly positive and negative) of the sentence, while ‘neu’ shows the overall percentage of neutral sentiments in the sentence and ‘neg’ shows the overall percentage of negative sentiments in the sentence. Fig.2 shows the confusion matrix that was calculated when we used Vader for sentiment analysis for our sample dataset and it achieved an accuracy of 88.31% ~88%.  
<img width="252" alt="image" src="https://user-images.githubusercontent.com/90365773/233864936-a9460d78-e0c0-419e-90c7-9a8983dcb73d.png">
Fig.2<br/>






# B.	Using TextBlob
TextBlob is a rich library that allows us to perform sentiment analysis and much more. TextBlob provides the polarity score and subjective score where if the polarity score ranges from -1.0 to 1.0, where -1.0 is very negative and 1.0 is very positive and the subjectivity score ranges from 0.0 to 1.0, where 0.0 is very objective and 1.0 is very subjective.
<img width="252" alt="image" src="https://user-images.githubusercontent.com/90365773/233864952-ebeb80cd-afb5-4abb-b86e-1349d51c30b3.png">
Fig.3<br/>

With TextBlob, we can also get the parts of speech tags and get the frequencies of the words in the text. We can also analyze the sentences in the corpus using NaiveBayes classifier or decisionTree classifier. Fig.3 shows the confusion matrix that was calculated when we used TextBlob for sentiment analysis for our sample dataset and it achieved an accuracy of 85.30% ~85.%.

# VI.	SENTIMENT ANALYSIS USING MACHINE LEARNING APPROACH

Machine learning approaches have been in trend for sentiment analysis for their ability to automatically improve from the data and improve accuracy overtime. In this approach we will be using Logistic Regression which is a supervised learning algorithm that is used to classify the data into discrete categories. For the case of Sentiment Analysis, the categories are normally classified as positive, negative, or neutral. Logistic Regression splits the data using a logistic function that allows probabilistic predictions of the category that the review belongs to. 

We begin by labelling the data into positive, negative, and neutral based on the ratings. The columns of reviewText and summary are concatenated into a new column called review_summary and any missing values are filled with empty strings. We used Regular Expressions (re) to remove any dollar signs ($), parenthesis (‘’), and non-word characters (*&^) from the review_summary column and convert the text into lower case. We used Tfidf Vectorizer to convert the review_summary column into a numerical feature representation, setting the max_features to 5000 and the resuling matrix is stored in bow_rep_tfidf variable. To balance the data set, we oversampled the minority classes. We split the data into training(70%) and testing(30%) and built two models namely: Multinomial Naïve Bayes and Logistic Regression. We fit the data and make predictions on the x_test and achieve an accuracy of 97.69% and 99.46% respectively.

# VII.	COMPARISION

There are some key differences between these two approaches that can be compared.
# A.	Accuracy
Machine learning approache can achieve higher accuracy for sentiment analysis task especially when trained on large labeled datasets and optimized for feature selection. While Lexicon based approaches are not as accurate as machine learning approaches for complex texts or in cases where the sentiment polarity is ambitus. 

# B.	Approach
Vader and TextBlob rely on prebuilt sentiments lexicons containing list of words with their associated sentiment scores. While Naïve Bayes and Logistic Regression use statistical models to learn the relationship between words and sentiment labels from labeled training data.

# C.	Training 
Vader and TextBlob are prebuilt sentiment lexicons that are already trained and hence do not require any additional training. However, they may require fine tuning for specific domains or languages. While Naïve Bayes and Logistic Regression require training on a labeled dataset, which can be time consuming and resource intensive. However, once strained the model can be used to classify sentiments for new data.

# D.	Application
Vader and TextBlob are well suited for sentiment classification for social media texts or product reviews where the sentiment polarity is often clear and the texts are relatively short while Naïve Bayes and Logistic Regression are better suited for tasks where the sentiment polarity may be more ambiguous and the text are longer such as analyzing sentiments in news articles or scientific papers.

# VIII.	CONCLUSION
In conclusion, both the Lexicon and Machine Learning approaches were effective in building sentiment analysis models for products based on customer reviews. However, the Random Forest Classifier model performed slightly better than the Lexicon models in terms of accuracy, precision, recall, and F1 score.

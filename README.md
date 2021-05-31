## Introduction:

Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

Challenge is  to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t. Access to a dataset of 10,000 tweets that were already classified. 

### Quick Look

Let's look at our data... first, an example of what is NOT a disaster tweet. Also, displaying the data set descriptive stats.
![](/image/Capture1.png)

### EDA

Plotting our target variable to get to know about the target distribution.
![](/image/target.png)


Dataset is not imbalanced, so there is no need for resampling techniques.

### Feature Engineering

Created three new features ,' hashtag','len','hashtag_count' and observe whether there is a relationship between target and new features.
![](/image/len_tweet.png)

![](/image/download.png)

We found a not so strong relationship between the variables, so we will not take this into our consideration.

### Building vectors

The theory behind the model we'll build in this notebook is pretty simple: the words contained in each tweet are a good indicator of whether they're about a real disaster or not (this is not entirely correct, but it's a great place to start).

We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.

Note: a vector is, in this context, a set of numbers that a machine learning model can work with. We'll look at one in just a second.
```python
count=CountVectorizer()
tf=TfidfVectorizer()
train_vectors=count.fit_transform(df1['text'][0:5])
train_vectors[0].todense()
```
The above tells us that:

1) There are 54 unique words (or "tokens") in the first five tweets.
2) The first tweet contains only some of those unique tokens - all of the non-zero counts above are the tokens that DO exist in the first tweet.

Now let's create vectors for all of our tweets

## Our model
As we mentioned above, we think the words contained in each tweet are a good indicator of whether they're about a real disaster or not. The presence of particular word (or set of words) in a tweet might link directly to whether or not that tweet is real.

What we're assuming here is a linear connection. So let's build a linear model and see!

Let's test our model and see how well it does on the training data. For this we'll use cross-validation - where we train on a portion of the known data, then validate it with the rest. If we do this several times (with different portions) we can get a good idea for how a particular model or method performs.

The metric for this competition is F1, so let's use that here

```python
clf=RidgeClassifier()
scores=cross_val_score(clf,train_vectors,df1['target'],cv=3,scoring='f1')
scores.mean()

```

The above scores aren't terrible! It looks like our assumption will score roughly 0.65 on the leaderboard. There are lots of ways to potentially improve on this (TFIDF, LSA, LSTM / RNNs, the list is long!) - give any of them a shot!

In the meantime, let's do predictions on our training set and build a submission for the competition.

```python
clf.fit(train_vectors,df1['target'])
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv") 
submission['target']=clf.predict(test_vectors)
submission.head()
```

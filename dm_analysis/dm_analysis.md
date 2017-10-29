
# Load Packages and Data


```python
import pandas as pd
import numpy as np
import nltk
import string
import re
import sys
import time
import matplotlib.pyplot as plt

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, pairwise_distances
from sklearn.cluster import KMeans

from xgboost import XGBRegressor
from skopt import gp_minimize
from skopt.plots import plot_convergence

%matplotlib inline
```


```python
df = pd.read_csv('dm_dat.csv')
print df.shape

# Drop obs with null value in title or pub_time
df = df[df['title'].notnull()]
print df.shape
df = df[df['pub_time'].notnull()]
print df.shape

# Reset index - much quicker to do it here than once I've made tf-idf matrix
df = df.reset_index()
```

    (54978, 14)
    (54977, 14)
    (54943, 14)


Let's explore the distributions of shares and comments, and drop outliers.


```python
print 'No shares:', sum(df['shares']==0)*100/float(len(df))
print 'No comments:', sum(df['comments']==0)*100/float(len(df))

print '>1500 shares:', sum(df['shares']>1500)*100/float(len(df))
print '>300 comments:', sum(df['comments']>300)*100/float(len(df))
print '>1500 shares or >300 comments:',sum((df['shares']>1500) | (df['comments']>300))*100/float(len(df))
df = df[(df['shares']<=1500) & (df['comments']<=300)]
df = df.reset_index()
```

    No shares: 57.3339642903
    No comments: 52.6764100977
    >1500 shares: 5.88792020822
    >300 comments: 5.03248821506
    >1500 shares or >300 comments: 8.89103252462



```python
plt.hist(df['shares'].tolist())
plt.xlabel("Shares")
plt.ylabel("Frequency")
plt.show()
plt.hist(df['comments'].tolist())
plt.xlabel("Comments")
plt.ylabel("Frequency")
plt.show()
```


![png](output_5_0.png)



![png](output_5_1.png)


# Preprocess Non-Text Features

Create dummy variables for key categories.


```python
df.category.value_counts().plot(kind='bar')
pop_cats = ['wires', 'news', 'tvshowbiz', 'sport', 'femail']
for c in pop_cats:
    df['cat_' + c] = 0
    df.loc[df.category==c, 'cat_' + c] = 1
```


![png](output_7_0.png)


# Preprocess Text Data

I need to convert the data from the titles into numerical features. To do this, I will create a TF-IDF matrix. In the process, I will in the following order (although note that there are multiple ways in which these operations could be combined):
<ol>
<li>**Remove all non-ASCII characters**: eg. \xe2\x80\x98</li>
<li>**Make all words lowercase**</li>
<li>**Remove punctuation**</li>
<li>***Tokenize***: divide string into a list of substrings.</li>
<li>**Remove words not containing letters**
<li>**Remove words containing numbers**
<li>**Remove stopwords**: stopwords are a list of high frequency words like, the, to, and also.</li>
<li>**Stem**: take the root of each word.</li>
<li>**Remove short words**: that is, of 3 characters or less.</li>
</ol>

I will carry out all but the first of these steps in the process of creating a TF-IDF matrix, below. But we need to prepare the functions beforehand.


```python
def no_punctuation_unicode(text):
    '''.translate only takes str, whereas TfidfVectorizer only takes unicode.
    Therefore, to use .translate in the tokenizer in TfidfVectorizer I need to
    write a function that converts unicode -> string, applies .translate,
    and then converts it back'''
    str_text = str(text)
    no_punctuation = str_text.translate(None, string.punctuation)
    unicode_text = no_punctuation.decode('utf-8')
    return unicode_text

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def prep_blurb(text):
    lowers = text.lower()
    no_punct = no_punctuation_unicode(lowers)
    tokens = nltk.word_tokenize(no_punct)
    has_letters = [t for t in tokens if re.search('[a-zA-Z]',t)]
    no_numbers  = [t for t in has_letters if not hasNumbers(t)]
    drop_stops = [w for w in no_numbers if not w in stoplist] 
    stems = [stemmer.stem(t) for t in drop_stops]
    drop_short = [s for s in stems if len(s)>2]
    return drop_short
```

I next extract a matrix of TF-IDF (term frequency-inverse document frequency) features.

The tf-idf weight of a term in a document is the product of its tf (term frequency) weight and its idf (inverse document frequency) weight. This is generally given by the formula:

$w_{t,d} = (1 + logtf_{t,d}) \times log_{10}(\frac{N}{df_t})$

Where the term frequency $tf_{t,d}$ of term t in document d is defined as the number of times that t occurs in d, the document frequency $df_t$ is he number of documents that contain t, and N is the number of documents. The computation of tf-idfs in scikit-learn is in fact [slightly different from the standard textbook notation](http://scikit-learn.org/dev/modules/feature_extraction.html#the-bag-of-words-representation), but we can ignore that here.

The tf-idf weight (of a term in a document) thus increases with the number of times a term occurs in a document, and also increases with the rarity of the term in the collection.

I will specify some key parameters I define in ```TfidfVectorizer```:
<ul>
<li> max_df: ignore terms that have a document frequency strictly higher than the given threshold (eg. ignore words that appear in more than .8 of documents).
<li> min_idf:  ignore terms that have a document frequency strictly lower than the given threshold (eg. ignore words that occure in less than .2 of the documents).
<li> norm: I use [L2 normalization](http://stats.stackexchange.com/questions/225564/scikit-learn-normalization-mode-l1-vs-l2-max) following an [scikit-learn example](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py) that I draw on later in my Lasso analysis. (I originally tried standardizing the dense datframe I created from the sparse matrix such that every predictor would have mean=0 and sd=1, following a different [Lasso example on Coursera](https://www.coursera.org/learn/machine-learning-data-analysis/home/welcome), but this was taking forever to run.)
</ul>

The following cell takes a while to run.


```python
# Get list of titles
tlist = df['title'].tolist()

##Remove all non-ASCII characters from titles
new_list = []
for t in tlist:
    new_list.append(t.decode('utf8').encode('ascii', errors='ignore'))
tlist = new_list
    
# Convert items in alist, as TfidfVectorizer only takes unicode
tlist = [t.decode('utf-8') for t in tlist]

# Stopwords must also be in unicode to work with TfidfVectorizer
stoplist = [word.decode('utf-8') for word in nltk.corpus.stopwords.words('english')] 

# Create stemmer
stemmer = SnowballStemmer("english")

# Don't use stop_words argument to TfidfVectorizer as delt with in prep_blurb
tfidf_vectorizer = TfidfVectorizer(tokenizer=prep_blurb, min_df=0.01, norm='l2')
%time tf_idf = tfidf_vectorizer.fit_transform(tlist)

print 'Length initial list:', len(tlist)
print 'tf_idf.shape:', tf_idf.shape
print 'type(tf_idf):', type(tf_idf)

list_of_words = tfidf_vectorizer.get_feature_names()
print 'List of words:', list_of_words[:50]
```

    CPU times: user 1min 5s, sys: 939 ms, total: 1min 6s
    Wall time: 1min 39s
    Length initial list: 50058
    tf_idf.shape: (50058, 104)
    type(tf_idf): <class 'scipy.sparse.csr.csr_matrix'>
    List of words: [u'ahead', u'arrest', u'attack', u'babi', u'back', u'ban', u'bank', u'boss', u'call', u'car', u'celebr', u'championship', u'charg', u'china', u'citi', u'claim', u'could', u'court', u'cup', u'cut', u'daughter', u'day', u'dead', u'deal', u'death', u'die', u'dress', u'end', u'england', u'face', u'famili', u'fan', u'fashion', u'fight', u'fire', u'first', u'forc', u'former', u'get', u'girl', u'give', u'head', u'help', u'high', u'hit', u'home', u'hous', u'husband', u'kill', u'latest']


Trying different values of min_df df and max_df it becomes clear that the vast majority of words appear in less than 1 percent of documents, while very few words appear in a high proportion of documents. I choose to ignore all words that appear in less than 1 percent of documents. 

Extracting tf-idf features, above, gives us a weight matrix between terms and documents. Each document is represented by a row of the matrix, which is a real valued vector, and each term by a column. This will be a sparse matrix, that is, a matrix in which most of the elements are zero.

scikit-learn's TfidfVectorizer returns a matrix in scipy.sparse.csr format. Printing this matrix shows the nonzero values of the matrix in the (row, col) format. 


```python
print 'tf_idf[:2,]:', tf_idf[:2,]
```

    tf_idf[:2,]:   (0, 74)	0.678596012457
      (0, 91)	0.73451170983
      (1, 75)	1.0


In order to include the features from my TF-IDF matrix in my regression model, I need to convert my scipy sparse matrix to a regular dataframe. 


```python
print tf_idf.shape
df_tfidf = pd.DataFrame(tf_idf.todense())
print tf_idf.shape
df_tfidf.columns = list_of_words
```

    (50058, 104)
    (50058, 104)


I next need to attach my tf-idf matrix dataframe to the other predictor variables. In Lasso regression, the penalty term is not fair if the predictor variables are not on the same scale, as this would mean that not all of the predictors get the same penalty. I normalized the variables in my tf-idf matrix already, and I now need to normalize the non-text variables (I originally tried standardizing, but this was extremely slow - see above discussion of paramaters in TfidfVectorizer).


```python
df_notext = df[['pub_days', 'pub_dayofweek', 'pub_sincemid', 'sponsored_dummy', 
                 'cat_wires', 'cat_news', 'cat_tvshowbiz', 'cat_sport', 'cat_femail']]
df_notext_1 = pd.DataFrame(normalize(df_notext, norm='l2', copy=False, axis=0))  #axis=0 normalizes each column
df_notext_1.columns = df_notext.columns.values
df_notext = df_notext_1
del df_notext_1
df_notext.head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pub_days</th>
      <th>pub_dayofweek</th>
      <th>pub_sincemid</th>
      <th>sponsored_dummy</th>
      <th>cat_wires</th>
      <th>cat_news</th>
      <th>cat_tvshowbiz</th>
      <th>cat_sport</th>
      <th>cat_femail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.003118</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.003416</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.003190</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



In joining my tf-idf dataframe with the other variables, concatenating and merging is extremely slow. The quickest approach seems to be to create an empty column for each variable I want to add, and then fill it with the appropriate values.


```python
for c in df_notext.columns.values.tolist():
    df_tfidf[c] = np.nan
    df_tfidf[c] = df_notext[c]
```

Finally, I need to make sure that I have joined each tf-idf row with the correct values of the other variables (ie. that I have joined all text features to the appropriate observation).

If I try selecting multiple columns from a row of df_tfidf, with a large dataframe it ends up sucking up loads of memory and can grind to a hault (this occured when I didn't restrict my tf-idf matrix to words that appeared at least one percent of the time). I believe that problem is that [pandas returns copies for most operations](http://stackoverflow.com/questions/25196595/memory-optimization-when-selecting-from-a-pandas-dataframe). Therefore, I'm better of selecting a row and then examing that. 


```python
print df.ix[0]['title']
test_row = df_tfidf.iloc[0]
test_row[test_row>0]
```

    Russia says Trump's efforts on Ukraine better than Obama's - TASS


    /Users/sambarrows/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.





    say              0.678596
    trump            0.734512
    pub_dayofweek    0.001371
    pub_sincemid     0.003118
    cat_wires        0.006175
    Name: 0, dtype: float64



## Lasso Regression

I employ Lasso Regression, as I have a large number of predictors. Lasso is a shrinkage method for linear regression models that selects the predictors to include. The Lasso imposes a constraint on the sum of the absolute values of the model parameters, where the sum has a specified constant as an upper bound. This contraint causes regression coefficients for some coefficients to shrink towards zero. This is the shrinkage process, which allows for better interpetation of the model and identifies the variables most strongly asssociated with the outcome of the model.

\begin{align*}
\hat{\beta}^{lasso} &= argmin_\beta\sum_{i=1}^N\big(y_i - \beta_0 - \sum_{j=1}^px_{ij}\beta_j\big)^2 \\
&s.t. \sum_{j=1}^p|\beta_j| \leq t
\end{align*}

Alternatively, we the model can be written as:

\begin{equation*}
\hat{\beta}^{lasso} = \Big\{argmin_\beta\frac{1}{2}\sum_{i=1}^N\big(y_i - \beta_0 - \sum_{j=1}^px_{ij}\beta_j\big)^2 + \lambda\sum_{j=1}^p|\beta_j|\Big\}
\end{equation*}

Below, we will use cross-validation to choose the penalty parameter, $\lambda$. I developed the code below from a [scikit-learn example](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html#sphx-glr-auto-examples-linear-model-plot-lasso-model-selection-py) and a [Coursera example](https://www.coursera.org/learn/machine-learning-data-analysis/lecture/0PTeX/data-management-for-lasso-regression-in-python).

Decide whether to look at shares or comments as outcome. Here I will report the results for shares, but the results for comments are similar.


```python
y = 'shares'
# y = 'comments'
```

Split the data into a training and test set. (With a larger dataframe, ```train_test_split``` makes the memory usage blow up. StackerOverflow has [instructions for getting around this](http://stackoverflow.com/questions/31467487/memory-efficient-way-to-split-large-numpy-array-into-train-and-test).)


```python
pred_train, pred_test, tar_train, tar_test = train_test_split(df_tfidf.as_matrix(), df[y].as_matrix(),
                                                              test_size=.2, random_state=123)
```

Specify the lasso regression model. I will use the least angle regression (LAR) algorithm to fit the model, and will use cross-validation to select the value of the penalty parameter. Note the scikit-learn calls this penalty parameter $\alpha$, but the more conventional term is $\lambda$.


```python
t1 = time.time()
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train, tar_train)
t_lasso_lars_cv = time.time() - t1
print 'train time (seconds):', t_lasso_lars_cv
```

    train time (seconds): 233.701644897


Plot mean square error for each fold.


```python
m_log_alphas = -np.log10(model.cv_alphas_)

plt.figure()
plt.plot(m_log_alphas, model.cv_mse_path_, ':')
plt.plot(m_log_alphas, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')
plt.show()
```

    /Users/sambarrows/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in log10
      """Entry point for launching an IPython kernel.
    /Users/sambarrows/anaconda2/lib/python2.7/site-packages/sklearn/utils/deprecation.py:75: DeprecationWarning: Function cv_mse_path_ is deprecated; Attribute ``cv_mse_path_`` is deprecated in 0.18 and will be removed in 0.20. Use ``mse_path_`` instead
      warnings.warn(msg, category=DeprecationWarning)
    /Users/sambarrows/anaconda2/lib/python2.7/site-packages/sklearn/utils/deprecation.py:75: DeprecationWarning: Function cv_mse_path_ is deprecated; Attribute ``cv_mse_path_`` is deprecated in 0.18 and will be removed in 0.20. Use ``mse_path_`` instead
      warnings.warn(msg, category=DeprecationWarning)



![png](output_29_1.png)


Print variable names and regression coefficients


```python
import operator
coefs = dict(zip(df_tfidf.columns, model.coef_))
coefs = { k:v for k, v in coefs.items() if v!=0 }
print 'Initial number coefs:', df_tfidf.shape[1]
print 'Coefs after fitting model:', len(coefs)
#sorted_coefs_abs = sorted(coefs.items(), key=lambda x:-abs(x[1]))
#sorted_coefs_abs
sorted_coefs = sorted(coefs.items(), key=lambda x:-x[1])
sorted_coefs[:20]
```

    Initial number coefs: 113
    Coefs after fitting model: 61





    [('cat_news', 6457.1089767406465),
     ('cat_sport', 2338.9548679840977),
     ('cat_femail', 2194.3944872910565),
     ('pub_dayofweek', 674.05328166285369),
     (u'fan', 67.344754789223174),
     (u'mother', 42.871103821602581),
     (u'result', 30.392230997152012),
     (u'die', 28.563209732591055),
     (u'could', 25.716163788412405),
     (u'leagu', 18.9138823326373),
     (u'day', 17.179499438430156),
     (u'citi', 14.415228738124005),
     (u'hit', 13.715704444636838),
     (u'dead', 12.913204844872839),
     (u'reveal', 11.143245395086296),
     (u'woman', 11.018106167140537),
     (u'take', 7.9141906048921165),
     (u'show', 7.6991464475261084),
     (u'kill', 6.6831222637888299),
     (u'first', 6.4862336765787969)]




```python
sorted_coefs[-20:]
```




    [(u'deal', -4.5588244435363281),
     (u'court', -5.6793259254127655),
     (u'win', -6.2947798170585978),
     (u'dress', -6.8887285606297972),
     (u'south', -7.0223257334073024),
     (u'share', -8.0188266404092978),
     (u'rise', -8.1114480059761789),
     (u'cut', -10.247472504081848),
     (u'bank', -10.6913551691437),
     (u'celebr', -12.020082541924978),
     (u'fashion', -12.176155312848051),
     (u'nation', -15.435111782995786),
     (u'husband', -18.054047338095565),
     (u'man', -20.018031968862449),
     (u'car', -28.865403158593868),
     (u'championship', -32.505194129965915),
     (u'england', -61.764827038362135),
     ('pub_days', -265.29318651599266),
     ('pub_sincemid', -1006.3057846629991),
     ('cat_wires', -13891.352753144492)]



How does model do on test data?


```python
def print_errors(model, pred_train, pred_test, tar_train, tar_test):
    # MSE from training and test data
    train_error = mean_squared_error(tar_train, model.predict(pred_train))
    test_error = mean_squared_error(tar_test, model.predict(pred_test))
    print 'Training data MSE:', train_error
    print 'Test data MSE:', test_error

    # MAE from training and test data
    train_error = mean_absolute_error(tar_train, model.predict(pred_train))
    test_error = mean_absolute_error(tar_test, model.predict(pred_test))
    print 'Training data MAE:', train_error
    print 'Test data MAE:', test_error
    
print_errors(model, pred_train, pred_test, tar_train, tar_test)
```

    Training data MSE: 30705.4470647
    Test data MSE: 33591.0307443
    Training data MAE: 76.7396327498
    Test data MAE: 79.9628524015


How large is this error in context of data?


```python
print min(df[y])
print max(df[y])
print np.std(df[y])
```

    0
    1500
    189.291839447


How does this compare to fitting model without text variables?


```python
pred_train, pred_test, tar_train, tar_test = train_test_split(df_notext.as_matrix(), df[y].as_matrix(),
                                                              test_size=.2, random_state=123)

t1 = time.time()
model=LassoLarsCV(cv=20, precompute=False).fit(pred_train, tar_train)
t_lasso_lars_cv = time.time() - t1
print 'train time (seconds):', t_lasso_lars_cv
print

# Number of coefs after fitting model
coefs = dict(zip(df_notext.columns, model.coef_))
print 'Initial number coefs:', df_notext.shape[1]
print 'Coefs after fitting model:', len(coefs)
print

# MSE and MAE
print_errors(model, pred_train, pred_test, tar_train, tar_test)
```

    train time (seconds): 5.94855499268
    
    Initial number coefs: 9
    Coefs after fitting model: 9
    
    Training data MSE: 30891.4842252
    Test data MSE: 33684.8088984
    Training data MAE: 76.1813939733
    Test data MAE: 79.182473179


This is rather disappointing. Including text features does not improve the predictive power of our model. This likely results, at least in part, from the fact that such a high proportion of words occur in less than 1 percent of the document titles. 

## Regression Using Features from K-Means

Previous research as found that clustering at the preprocessing stage can help to reduce predictive error; for example, [Trivedi, Pardos, and Heffernan (2015)](https://arxiv.org/pdf/1509.06163.pdf). If this worked in this context, it would have the added advantage of allowing us to more meaninfully interpret our model; for example, we might be able to say that articles discussing soccer victories are particularly widely shared.

I will use k-means to cluster articles at the pre-processing stage, and then include these clusters as predictors in a regression model. Starting with k cluster centers, k-means identifies which points are nearest to each cluster center. It then chooses new cluster centers by minimizing the total quadratic distance of each cluster center to its points. Then it repeats this process, starting with these new cluster centers.

My dataframe is already to some extent grouped by topic according to the category variable. I will use k-means clustering for only the two largest of these groups: news and wires. I therefore need to attach an id variable to my dataframe so that I don't loose track of these observations, and then take only the news and wires stories. Note that non of the following code is going to work, absent a huge amount of RAM, with a very big dataset.


```python
df_tfidf['article_id'] = np.nan
df_tfidf['article_id'] = df['article_id']
```


```python
df_tfidf.columns.values
```




    array([u'ahead', u'arrest', u'attack', u'babi', u'back', u'ban', u'bank',
           u'boss', u'call', u'car', u'celebr', u'championship', u'charg',
           u'china', u'citi', u'claim', u'could', u'court', u'cup', u'cut',
           u'daughter', u'day', u'dead', u'deal', u'death', u'die', u'dress',
           u'end', u'england', u'face', u'famili', u'fan', u'fashion',
           u'fight', u'fire', u'first', u'forc', u'former', u'get', u'girl',
           u'give', u'head', u'help', u'high', u'hit', u'home', u'hous',
           u'husband', u'kill', u'latest', u'lead', u'leagu', u'leav', u'life',
           u'like', u'look', u'love', u'make', u'man', u'meet', u'mother',
           u'nation', u'new', u'one', u'open', u'parti', u'plan', u'polic',
           u'presid', u'report', u'result', u'return', u'reveal', u'rise',
           u'say', u'see', u'set', u'share', u'show', u'son', u'south',
           u'stand', u'star', u'state', u'take', u'talk', u'test', u'three',
           u'time', u'top', u'tri', u'trump', u'two', u'use', u'want', u'warn',
           u'week', u'white', u'win', u'woman', u'women', u'work', u'world',
           u'year', 'pub_days', 'pub_dayofweek', 'pub_sincemid',
           'sponsored_dummy', 'cat_wires', 'cat_news', 'cat_tvshowbiz',
           'cat_sport', 'cat_femail', 'article_id'], dtype=object)




```python
df_1 = df_tfidf[(df_tfidf['cat_news']>0) | (df_tfidf['cat_wires']>0)]
```

I next need to drop non-text variables from dataframe I will use for clustering, as I only want to cluster by text features. Also drop the id variable, but keep this vector for later.


```python
df_1 = df_1.drop(df_notext.columns.values.tolist(), axis=1)   # get warning if do it inplace
df_1_id = df_1['article_id']
df_1.drop('article_id', axis=1, inplace=True)
df_1.columns.values
```




    array([u'ahead', u'arrest', u'attack', u'babi', u'back', u'ban', u'bank',
           u'boss', u'call', u'car', u'celebr', u'championship', u'charg',
           u'china', u'citi', u'claim', u'could', u'court', u'cup', u'cut',
           u'daughter', u'day', u'dead', u'deal', u'death', u'die', u'dress',
           u'end', u'england', u'face', u'famili', u'fan', u'fashion',
           u'fight', u'fire', u'first', u'forc', u'former', u'get', u'girl',
           u'give', u'head', u'help', u'high', u'hit', u'home', u'hous',
           u'husband', u'kill', u'latest', u'lead', u'leagu', u'leav', u'life',
           u'like', u'look', u'love', u'make', u'man', u'meet', u'mother',
           u'nation', u'new', u'one', u'open', u'parti', u'plan', u'polic',
           u'presid', u'report', u'result', u'return', u'reveal', u'rise',
           u'say', u'see', u'set', u'share', u'show', u'son', u'south',
           u'stand', u'star', u'state', u'take', u'talk', u'test', u'three',
           u'time', u'top', u'tri', u'trump', u'two', u'use', u'want', u'warn',
           u'week', u'white', u'win', u'woman', u'women', u'work', u'world',
           u'year'], dtype=object)



Then fit model. I tried different numbers of clusters, 6 clusters gives the most meaningful groupings.


```python
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
%time km = kmeans.fit(df_1)
```

    CPU times: user 8.19 s, sys: 759 ms, total: 8.95 s
    Wall time: 11.6 s


I next want to look at some of the text in my different clusters to see if it seems reasonable. 

In a good clustering of documents:
* Documents in the same cluster should be similar.
* Documents from different clusters should be less similar.

To examine the text in my clustering I will:
* Fetch 5 nearest neighbors of each centroid from the set of documents assigned to that cluster. I will consider these documents as being representative of the cluster.
* Print top 5 words that have highest tf-idf weights in each centroid.


```python
centroids = km.cluster_centers_   # coordinates of cluster centers
cluster_assignment = km.labels_   # label of every data point

for c in xrange(n_clusters):
    # Cluster heading
    print('Cluster {0:d}    '.format(c))

    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroids[c].argsort()[::-1]
    for i in xrange(5):
        print list_of_words[idx[i]], centroids[c,idx[i]]
    print ('')
    
    # Compute distances from the centroid to all data points in the cluster
    distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
    distances[cluster_assignment!=c] = float('inf') # remove non-members from consideration
    nearest_neighbors = distances.argsort() # argsort() returns the indices that would sort an array
    # For 5 nearest neighbors, print the first 250 characters of text
    for i in xrange(5):
        print tlist[nearest_neighbors[i]][:250]
        print ('')
    print('==========================================================')
```

    Cluster 0    
    new 0.676215639123
    south 0.0329973543237
    say 0.0268965803513
    deal 0.0241164035692
    home 0.0191492984279
    
    PSA unveils new DS7 Crossback sport utility vehicle
    
    Katy Perry rocks vibrant pink suit in promo shots for her new single... after admitting she has 'given up' caring what people think about her style
    
    Kristen Stewart jets out of LA after creating a real buzz with her new shaved hairdo
    
    The Talented Mr Damon: Boston native Matt will narrate a new documentary about the marathon bombing in his hometown
    
    European rights official denounces new Hungarian asylum law
    
    ==========================================================
    Cluster 1    
    win 0.0212595767793
    china 0.019911651449
    polic 0.0199069152434
    top 0.0144398629067
    back 0.014095267931
    
    Hoges: The Paul Hogan Story rates behind music biopics INXS and Molly with 1.32M viewers
    
    Fraudster posed as Nickelback's drummer in a bid to swindle $25,000 worth of high-end microphones
    
    Ex-deputy accused of sexually assaulting female inmates
    
    Turkey's treatment of journalist damaging German-Turkish ties-minister
    
    Judge tosses part of ex-University of Kansas rower's lawsuit
    
    ==========================================================
    Cluster 2    
    trump 0.676416904556
    say 0.0578515938049
    ban 0.0497506672712
    presid 0.0445104052775
    hous 0.0366790495898
    
    Fact check: How Trumps Keystone XL story fell apart
    
    Trump dismisses dozens of Obama-era federal prosecutors
    
    U.S. senators ask government for proof Obama wiretapped Trump
    
    US STOCKS-Wall St slips as Trump speech looms; retail a drag
    
    Watchdog presses U.S. lawmakers to probe Icahn's role with Trump
    
    ==========================================================
    Cluster 3    
    man 0.62508739255
    polic 0.0696424641163
    charg 0.0508542814007
    kill 0.0447398528037
    arrest 0.0399346478452
    
    He's all bent out of shape! Hypermobile man rotates his hand a full 360 DEGREES (and squeamish viewers will find it weirdly unsettling)
    
    Man, 19, crushed by a wall while rebuilding a town destroyed by bushfires in Western Australia
    
    Swiss man suspected of abusing 80 boys in Thailand
    
    Poland confirms Minnesota man as Nazi commander
    
    James McAvoy is a man in black at SXSW screening of action spy thriller Atomic Blonde
    
    ==========================================================
    Cluster 4    
    hit 0.436382782225
    high 0.37478478732
    new 0.0299854903405
    rise 0.0288243799142
    share 0.0222457176637
    
    FOREX-Dollar index hits 5-day high after strong U.S. private-sector jobs data
    
    The number of smokers plunges to just 17 per cent of people as those quitting hits a 40-year high
    
    McCartney hits high, pays tribute to George Michael in Paris
    
    High alcohol prices at Australian pubs and clubs causing two-thirds of revellers to pre-drink before they hit the town
    
    U.S. auto recalls hit record high 53.2 million in 2016
    
    ==========================================================
    Cluster 5    
    say 0.709512773112
    china 0.0293953760962
    deal 0.0194353971958
    polic 0.0176372707732
    latest 0.0165136686419
    
    Vietnam's PM says ready to visit U.S. to promote ties
    
    U.S. military says deploys THAAD anti-missile defence system in S.Korea
    
    Greek PM glosses over delays and weak data, says economy is recovering
    
    Sturgeon says autumn 2018 would be "common sense" date for independence referendum - BBC
    
    GLOBAL ECONOMY-How do you say deja vu in Greek?
    
    ==========================================================
    Cluster 6    
    kill 0.702987348849
    attack 0.0470190919804
    polic 0.0460776236312
    two 0.0424183808647
    say 0.0371749468951
    
    Blasts at Serbia ammunition depot; 1 killed, some 25 injured
    
    At least 16 killed in Panama after bus drives into ravine
    
    Twin suicide bombing kills 26 at a wedding north of Baghdad
    
    'Jihadists' kill civilians, soldiers in north Mali
    
    Bear killed in German zoo after it escapes from compound
    
    ==========================================================
    Cluster 7    
    talk 0.78780394226
    say 0.0592039032491
    trump 0.0562192430682
    meet 0.0192247152431
    high 0.0163419757889
    


    /Users/sambarrows/anaconda2/lib/python2.7/site-packages/ipykernel_launcher.py:16: VisibleDeprecationWarning: boolean index did not match indexed array along dimension 0; dimension is 50058 but corresponding boolean dimension is 34299
      app.launch_new_instance()


    Silence is golden - May plays down Brexit talk at EU summit
    
    Talks continue over Ryder Cup-style multi-sport showdown between Great Britain and the USA
    
    Russia in touch with Syrian opposition rebels who boycotted talks - Ifax
    
    ASEAN, EU agree to lay groundwork to resume free trade talks
    
    'Why choose burger over prime steak?' TOWIE's Amber Turner locks horns with beau Jamie Reed in furious crisis talks... as she's SLAMMED for Dan Edgar cheating scandal
    
    ==========================================================
    Cluster 8    
    result 0.520059541699
    stand 0.498070731649
    championship 0.316426107476
    leagu 0.0259366561748
    cup 0.0181209753428
    
    Greek championship results and standings
    
    Bulgarian championship results and standings
    
    Soccer-Bolivian championship adecuacion results and standings
    
    Peruvian championship results and standings
    
    Soccer-Colombian championship liga aguila i results and standings
    
    ==========================================================
    Cluster 9    
    year 0.621488845268
    two 0.0510183884721
    three 0.0396204619145
    get 0.0327897030788
    first 0.0289398983572
    
    Coat check! As Emily Ratajkowski continues to rock last year's biggest outerwear trend, FEMAIL rounds up 24 stylish duster coats for YOU to shop
    
    Poland may need to consider rate hike this year - central banker
    
    Defense lawyer still going strong at 94 years old
    
    Patient, 39, who dubbed himself the god of explosives before posting home-made bombs through his GP surgerys letterbox in a row over prescriptions is jailed for five years
    
    Google and Levi's 280 smart jacket that lets you control your phone by brushing your sleeve will go on sale this year
    
    ==========================================================


These clusterings are far from perfect. For one thing, some clusterings appear to be driven by a word that is in fact not particularly substantively informative, such as "new", "man", or "years." If I explore using k-means further, I could address this by adding customer stopwords. For now, I will include all the categories it gives me, as lasso should end up dropping them if they are not helpful predictors.  

An additional concern is that several clusterings appear to be driven by a single word; however, given the relatively small number of words that appear in amore than 1 percent of the article titles, this may be unavoidable. Others are not driven by a single word, but are also not completely consistent in the substantive topics they put together. Thus, the current groupings should be considered a rough-and-ready first cut.

I next create a categorical variable giving the name of the cluster for each observation.


```python
clusters = km.labels_
km_clusters = ['new', 'legal', 'trump', 'man', 'high', 'glob_pol', 
               'killing', 'talks', 'championship', 'years']
for i in range(len(km_clusters)):
    clusters = [km_clusters[i] if x==i else x for x in clusters]
clusters = clusters + [None] * (len(df_notext) - len(clusters))     # create empty elements for obs not in subsample
```

Create dataframe with dummy variable for each of the categories and normalize.


```python
df_km = pd.DataFrame({'clusters': clusters})
for c in km_clusters:
    df_km[c] = 0
    df_km.loc[df_km.clusters==c, c] = 1
df_km.drop('clusters', axis=1, inplace=True)
df_km = pd.DataFrame(normalize(df_km, norm='l2', copy=False, axis=0))  # nb. copy doesn't make it inplace
df_km.columns = km_clusters
df_km.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new</th>
      <th>legal</th>
      <th>trump</th>
      <th>man</th>
      <th>high</th>
      <th>glob_pol</th>
      <th>killing</th>
      <th>talks</th>
      <th>championship</th>
      <th>years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022582</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.006365</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.024581</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.030289</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.034239</td>
    </tr>
  </tbody>
</table>
</div>



Merge with ```df_notext``` dataframe.


```python
for c in df_notext.columns.values.tolist():
    df_km[c] = np.nan
    df_km[c] = df_notext[c]
# df_km.drop(['cat_wires', 'cat_news'], axis=1, inplace=True)   # made model perform a lot worse
df_km.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new</th>
      <th>legal</th>
      <th>trump</th>
      <th>man</th>
      <th>high</th>
      <th>glob_pol</th>
      <th>killing</th>
      <th>talks</th>
      <th>championship</th>
      <th>years</th>
      <th>pub_days</th>
      <th>pub_dayofweek</th>
      <th>pub_sincemid</th>
      <th>sponsored_dummy</th>
      <th>cat_wires</th>
      <th>cat_news</th>
      <th>cat_tvshowbiz</th>
      <th>cat_sport</th>
      <th>cat_femail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022582</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.003118</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.006365</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.003416</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.024581</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.003190</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.030289</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.005972</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.034239</td>
      <td>0.0</td>
      <td>0.001371</td>
      <td>0.003454</td>
      <td>0.0</td>
      <td>0.006175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Fit regression model using same approach as above, to ensure comparable.


```python
pred_train, pred_test, tar_train, tar_test = train_test_split(df_km.as_matrix(), df[y].as_matrix(),
                                                              test_size=.2, random_state=123)

t1 = time.time()
model=LassoLarsCV(cv=20, precompute=False).fit(pred_train, tar_train)
t_lasso_lars_cv = time.time() - t1
print 'train time (seconds):', t_lasso_lars_cv
print

# Number of coefs after fitting model
coefs = dict(zip(df_notext.columns, model.coef_))
print 'Initial number coefs:', df_notext.shape[1]
print 'Coefs after fitting model:', len(coefs)
print

# MSE and MAE
print_errors(model, pred_train, pred_test, tar_train, tar_test)
```

    train time (seconds): 24.1776878834
    
    Initial number coefs: 9
    Coefs after fitting model: 9
    
    Training data MSE: 30884.8079964
    Test data MSE: 33699.8498576
    Training data MAE: 76.2085491638
    Test data MAE: 79.2159466417


Again, including text data fails to improve the predictive power of the model.

## XGBoost


```python
# split into test and training data (I'll go back to using original data, without features from k-means)
X_train, X_test, y_train, y_test = train_test_split(df_tfidf.as_matrix(), df[y].as_matrix(),
                                                    test_size=.2, random_state=123)
```


```python
# fit the model with defaults
t1 = time.time()
model = XGBRegressor().fit(X_train, y_train)
t = time.time() - t1
print 'train time (seconds):', t

# make and evaluate predictions
print_errors(model, pred_train, pred_test, tar_train, tar_test)
```

    train time (seconds): 16.7377119064
    Training data MSE: 29348.6166602
    Test data MSE: 33580.7312067
    Training data MAE: 74.5390734826
    Test data MAE: 79.2257482857


### Optimize hyperparameters


```python
# define objective function we want to minimize 
def objective(params):
    max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree = params

    model.set_params(max_depth=max_depth, 
                     learning_rate=learning_rate, 
                     n_estimators=n_estimators, 
                     gamma=gamma,
                     min_child_weight=min_child_weight,
                     subsample=subsample, 
                     colsample_bytree=colsample_bytree)

    return -np.mean(cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_error'))

# define bounds of the dimensions of the search space we want to explore
space = [(4, 12),             # max_depth
         (0.01, 0.5),         # learning_rate
         (20, 100),           # n_estimators
         (0, 0.5),            # gamma
         (1, 5),              # min_child_weight
         (0.1, 1),            # subsample
         (0.1, 1)]            # colsample_bytree
```


```python
model = XGBRegressor()
t1 = time.time()
res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)
t2 = time.time() - t1

print 'Time:', t2
"Best score=%.4f" % res_gp.fun
```

    Time: 2290.87547112





    'Best score=30651.1598'




```python
def best_params(res_gp):
    print("""Best parameters:
    - max_depth=%d
    - learning_rate=%.6f
    - n_esimators=%.6f
    - gamma=%.6f
    - min_child_weight=%.6f
    - subsample=%.6f
    - colsample_bytree=%.6f""" % (res_gp.x[0], 
                                  res_gp.x[1], 
                                  res_gp.x[2], 
                                  res_gp.x[3], 
                                  res_gp.x[4], 
                                  res_gp.x[5],
                                  res_gp.x[6]))
best_params(res_gp)
```

    Best parameters:
        - max_depth=4
        - learning_rate=0.122313
        - n_esimators=36.000000
        - gamma=0.000000
        - min_child_weight=1.000000
        - subsample=0.916504
        - colsample_bytree=0.668645



```python
plot_convergence(res_gp)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1165b2950>




![png](output_66_1.png)



```python
model_opt = XGBRegressor(max_depth=res_gp.x[0], 
                         learning_rate=res_gp.x[1], 
                         n_estimators=res_gp.x[2],
                         gamma=res_gp.x[3], 
                         min_child_weight=res_gp.x[4],
                         subsample=res_gp.x[5],
                         colsample_bytree=res_gp.x[6]).fit(X_train, y_train)
print_errors(model_opt, pred_train, pred_test, tar_train, tar_test)
```

    Training data MSE: 29309.6756115
    Test data MSE: 33422.2739789
    Training data MAE: 73.4934628747
    Test data MAE: 77.9925192076


How does this compare to fitting model without text variables?


```python
X_train, X_test, y_train, y_test = train_test_split(df_notext.as_matrix(), df[y].as_matrix(),
                                                    test_size=.2, random_state=123)

model = XGBRegressor()
res_gp_notext = gp_minimize(objective, space, n_calls=100, random_state=0)

print best_params(res_gp)
plot_convergence(res_gp)
```

    Best parameters:
        - max_depth=4
        - learning_rate=0.122313
        - n_esimators=36.000000
        - gamma=0.000000
        - min_child_weight=1.000000
        - subsample=0.916504
        - colsample_bytree=0.668645
    None





    <matplotlib.axes._subplots.AxesSubplot at 0x11932eb10>




![png](output_69_2.png)



```python
model_opt_notext = XGBRegressor(max_depth=res_gp.x[0], 
                         learning_rate=res_gp.x[1], 
                         n_estimators=res_gp.x[2],
                         gamma=res_gp.x[3], 
                         min_child_weight=res_gp.x[4],
                         subsample=res_gp.x[5],
                         colsample_bytree=res_gp.x[6]).fit(X_train, y_train)
print_errors(model_opt_notext, X_train, X_test, y_train, y_test)
```

    Training data MSE: 30246.1536175
    Test data MSE: 33340.4866103
    Training data MAE: 74.3490005988
    Test data MAE: 77.7349067091


XGBoost is a slight improvement on regression. Again, however, including features derived from article headlines does not improve upon a model that includes only non-text features.

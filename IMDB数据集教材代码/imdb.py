#忽略一些因评论为类似路径的文本而引起的错误
import warnings
from bs4 import MarkupResemblesLocatorWarning
from streamlit.string_util import clean_text
# 在文件开头添加
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
def review_to_word(raw_review):
    # 现在不会显示警告
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    # ... 其他处理
    return clean_text

import pandas as pd
file_path = r'D:\360Downloads\labeledTrainData.tsv'
train = pd.read_csv(file_path, header=0,delimiter="\t", quoting=3)
from bs4 import BeautifulSoup
example1 = BeautifulSoup(train["review"][0], "lxml")
print(train["review"][0])                           #输出原始文本
print(example1.get_text())                          #输出的文本中标签或标记被删除了
import re
letters_only = re.sub("[^a-zA-Z]",         # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )       # The text to search
print(letters_only)                                 #删除了标点符号，只保留了字母
lower_case = letters_only.lower()        # 小写化
words = lower_case.split()               # Split into words
import nltk
#nltk.download('stopwords')                          # 下载数据集，包括停止词，and，is，the等
from nltk.corpus import stopwords        # 导入停止词列表
print(stopwords.words("english"))
words = [w for w in words if not w in stopwords.words("english")]
print(words)
def review_to_words( raw_review ):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text,)
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)

clean_review = review_to_words( train["review"][0] )
print(clean_review)

#循环清理所有训练集
# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
print("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if (i + 1)%1000 == 0:
        print("Review %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append( review_to_words( train["review"][i] ))
#循环清理结束
print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer
#创建词袋
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()
#查看训练数据数组现在的外观
print(train_data_features.shape)
# 查看词汇表
vocab = vectorizer.get_feature_names_out()
print(vocab)
#打印词汇表
import numpy as np
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)
#随机森林算法
print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees    分100棵树
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )


#测试数据
# Read the test data
testFile_path = r'D:\360Downloads\testData.tsv'
test = pd.read_csv(testFile_path, header=0, delimiter="\t",
                   quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if (i + 1) % 1000 == 0:
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
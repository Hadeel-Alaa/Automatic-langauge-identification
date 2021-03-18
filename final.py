# import all required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# read the dataset
df = pd.read_csv("AI_dataset.csv")

#start preprocessing
documents = df['Text'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

# Using the elbow method to find the optimal number of clusters
#wcss = []
#for i in range(1, 30):
#   kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42,max_iter=800)
#   kmeans.fit(features)
#   wcss.append(kmeans.inertia_)#Thus is the WCSS of the trained clustering mdoel
#plt.plot(range(1, 11), wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')#Within Cluster sum of square
#plt.show()

#start K-means
k = 22
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(features)
df['cluster'] = model.labels_

# to print the first five rows to check clusters
#print(df.head())


Xfeatures =df['Text']

# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

# Features 
X
# Labels
y = df.language

# training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# the accuracy of test set of the Model
#print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

# accuracy of training set of the Model
#print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")

#scan input from user
val = input("Enter your sentence: ") 

# Sample1 Prediction
sample_name = [val]
vect = cv.transform(sample_name).toarray()
print("The launguage is:", clf.predict(vect))
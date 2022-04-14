import pandas as pd
ds=pd.read_csv('E:dd.csv', sep=';')
print (ds)

# Importing LabelEncoder from Sklearn
# library from preprocessing Module.
from sklearn.preprocessing import LabelEncoder

# Creating a instance of label Encoder.
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(ds['gender'])

# printing label
print("laaaaaaaaaaaaaaaaaaaaaaaaabeeelllllllllllllllllllll")
#print(label)
# removing the column 'gender' from df
# as it is of no use now.
ds.drop("gender", axis=1, inplace=True)

# Appending the array to our dataFrame

ds.insert(1, 'gender', label)

# printing Dataframe
print("newwwwwwwwwwwds")
print(ds)

X=ds.iloc[:,:-1]
y=ds.iloc[:,-1]
print("xxxxxxxxxxxxxxxxxxxxx")
print(X)
print("yyyyyyyyyyyyyyyyyyyy")
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print('accuracy')
print(ac)
print(cm)

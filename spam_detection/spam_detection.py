import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

import seaborn as sns 



# read in data 
df = pd.read_csv('spam.csv')

# Dataset consists of two categories: ham and spam
# Replacing the the categories with numbers 
#preprocessing of data
labels = {"ham":0,
          "spam":1}

df['Category'] = df["Category"].apply(lambda x : labels[x])


# define input and output of model 
x = df['Message']
y = df['Category']


# train test split
# 70 % training
# 30 % testing 
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size = 0.3,
                                                 random_state = 42)




# Naive Bayes will be used
# initilize count vectorizer object
# counts the words in every message
# each word is in a column, and the entries are how many times the word occures in one message
vectorizer = CountVectorizer()



# applying the count vectorizer 
x_train_vectorizer = vectorizer.fit_transform(x_train)



# show the train data in a more readable form 
df2 = pd.DataFrame(data = x_train_vectorizer.toarray(),
                   columns=vectorizer.get_feature_names_out())

df2 = df2.astype(float)



# load the model 
model = MultinomialNB(fit_prior = True)


# train the model 
model.fit(x_train_vectorizer,y_train)


# testing the model 
x_test_vectorizer = vectorizer.transform(x_test)
y_pred = model.predict(x_test_vectorizer)




# get accuracy 
accuracyscore = accuracy_score(y_test,y_pred)

# get precision and recall and f1 scores for each class 
print(classification_report(y_test, y_pred))


# plot the confusion matrix 
cm = confusion_matrix(y_test,y_pred)

s = sns.heatmap(cm,
                annot = True,
                xticklabels = list(labels.keys()),
                yticklabels = list(labels.keys()))
            
s.set(xlabel='Predicted', ylabel='True')







# try out by inputing a certain message and see how it performs 
vector  = ["You have won a free TV"]

# convert 
vector_meassage= vectorizer.transform(vector)

# predict 
pred2 = model.predict(vector_meassage)

# print out prediction result
for k,v in labels.items(): 
    
    if v == pred2: 
        
        print("The model predicts :" + vector[0] + " is a " + k + " message")



# get the probabilties for each class 
pred2_prob = model.predict_proba(vector_meassage)
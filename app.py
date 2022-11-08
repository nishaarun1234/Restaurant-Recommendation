# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 19:39:19 2022

@author: arunk
"""


import pandas as pd
import numpy as np
from flask import Flask, render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import Dataset 
restraurant = pd.read_excel("C:/Users/arunk/OneDrive/Desktop/Last project/new dataset.xlsx")
restraurant.shape # shape
restraurant.columns




### Identify duplicates records in the data ###
duplicate = restraurant.duplicated()
duplicate
sum(duplicate)  ## no duplicate datas found



# check for null values
restraurant.isna().sum()

from sklearn.impute import SimpleImputer


# # Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

restraurant["Email_id"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["Email_id"]]))

restraurant["State"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["State"]]))

restraurant["City"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["City"]]))

restraurant["Frequency of Visit"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["Frequency of Visit"]]))

restraurant["Preferred Meal"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["Preferred Meal"]]))

restraurant["Preferred Category (Veg or Non-Veg)"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["Preferred Category (Veg or Non-Veg)"]]))

restraurant["order1"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["order1"]]))

restraurant["order2"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["order2"]]))

restraurant["order3"] = pd.DataFrame(mode_imputer.fit_transform(restraurant[["order3"]]))

restraurant.isnull().sum()  

# Mean Imputer for Numerical Data 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
restraurant["rating"] = pd.DataFrame(mean_imputer.fit_transform(restraurant[["rating"]]))
restraurant["rating"].isna().sum()

restraurant.rating = restraurant.rating.astype('int64')


# Changing the Dataframe name to anime
food = restraurant

# Checking the shape (showing the number of rows & column)
food.shape 

# Checking the column name (showing the column name)
food.columns

#term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer 

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# Preparing the Tfidf matrix by fitting and transforming
#Initiated all possible criteria based on Customer Details which be using to arrive the results.

tfidf_matrix = tfidf.fit_transform(food['Preferred Meal']+food['Preferred Category (Veg or Non-Veg)']+food['Email_id']+food['Frequency of Visit']+food['MainIngredients1'])
tfidf_matrix1 = tfidf.fit_transform(food['Preferred Meal']+food['Preferred Category (Veg or Non-Veg)']+food['Email_id']+food['Frequency of Visit']+food['MainIngredients2'])
tfidf_matrix2 = tfidf.fit_transform(food['Preferred Meal']+food['Preferred Category (Veg or Non-Veg)']+food['Email_id']+food['Frequency of Visit']+food['MainIngredients3'])


#importing the linear kernel library
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_matrix1 = linear_kernel(tfidf_matrix1, tfidf_matrix1)
cosine_sim_matrix2 = linear_kernel(tfidf_matrix2, tfidf_matrix2)


# creating a mapping of anime name to index number 
food_index = pd.Series(food.index, index = food['order1']).drop_duplicates()
food_index1 = pd.Series(food.index, index = food['order2']).drop_duplicates()
food_index2 = pd.Series(food.index, index = food['order3']).drop_duplicates()



def get_recommendations(order1, topN):
    food_id = food_index[order1]
    global result
    cosine_scores = list(enumerate(cosine_sim_matrix[food_id]))
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    food_idx  =  [i[0] for i in cosine_scores_N]
    food_scores =  [i[1] for i in cosine_scores_N]
    result = pd.DataFrame(columns=["order1", "Score"])
    result["order1"] = food.loc[food_idx, "order1"]
    result["Score"] = food_scores
    result.reset_index(inplace = True)  
 # food_similar_show.drop(["index"], axis=1, inplace=True)
    print (result)
    return (result)

get_recommendations("Pongal", topN = 5)

# If we use the same order on order2 & Order3, the recommendation will be different, will not provide the same output.
# IF we recommend the same item it would not be correct, hence the suggestion will be different based on the order items.


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/about',methods=['POST'])
def getvalue():
    Ordername = request.form['Ordername']
    get_recommendations(Ordername, topN = 10)
    dfs=result
    return render_template('result.html',  tables=[dfs.to_html(classes='data')], titles=dfs.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
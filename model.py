# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:47:48 2022

@author: arunk
"""



########   Restaurant Recommendation Engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


restraurant = pd.read_excel("C:/Users/arunk/OneDrive/Desktop/Last project/new dataset.xlsx")
#restraurant.shape
#restraurant.columns



### Identify duplicates records in the data ###
#duplicate = restraurant.duplicated()
#duplicate
#sum(duplicate)  ## no duplicate datas found



# check for null values
#restraurant.isna().sum()

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
#food.shape 

# Checking the column name (showing the column name)
#food.columns

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

#Checking whether the below code shows the column index no.
food_id = food_index["Dosa"]
food_id


def get_recommendations(order1, topN):    
    # topN = 10
    # Getting the order1 index using its title 
    food_id = food_index[order1]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[food_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar Order1 
    cosine_scores_N = cosine_scores[0: topN+1]
    

# Getting the Order1 index 
    food_idx  =  [i[0] for i in cosine_scores_N]
    food_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar Order1 and scores
    food_similar_show = pd.DataFrame(columns=["order1", "Score"])
    food_similar_show["order1"] = food.loc[food_idx, "order1"]
    food_similar_show["Score"] = food_scores
    food_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    #print (food_similar_show)
    return (food_similar_show)


# defing the recommendation function based on the Order Item2

def get_recommendations1(order2, topN):    
    # topN = 10
    # Getting the Order2 index using its title 
    food_id1 = food_index1[order2]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # food
    cosine_scores1 = list(enumerate(cosine_sim_matrix1[food_id1]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores1 = sorted(cosine_scores1, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar Order2 
    cosine_scores_N1 = cosine_scores1[0: topN+1]
    
    # Getting the Order2 index 
    food_idx1  =  [i[0] for i in cosine_scores_N1]
    food_scores1 =  [i[1] for i in cosine_scores_N1]
    
    # Similar Order2 and scores
    food_similar_show1 = pd.DataFrame(columns=["order2", "Score"])
    food_similar_show1["order2"] = food.loc[food_idx1, "order2"]
    food_similar_show1["Score"] = food_scores1
    food_similar_show1.reset_index(inplace = True)  
    # food_similar_show.drop(["index"], axis=1, inplace=True)
    print (food_similar_show1)
    # return (food_similar_show)

# defing the recommendation function based on the Order Item3

def get_recommendations2(order3, topN):    
    # topN = 10
    # Getting the Order3 index using its title 
    food_id2 = food_index2[order3]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores2 = list(enumerate(cosine_sim_matrix2[food_id2]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores2 = sorted(cosine_scores2, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar Order3 
    cosine_scores_N2 = cosine_scores2[0: topN+1]
    
    # Getting the movie index 
    food_idx2  =  [i[0] for i in cosine_scores_N2]
    food_scores2 =  [i[1] for i in cosine_scores_N2]
    
    # Similar Order3 and scores
    food_similar_show2 = pd.DataFrame(columns=["order3", "Score"])
    food_similar_show2["order3"] = food.loc[food_idx2, "order3"]
    food_similar_show2["Score"] = food_scores2
    food_similar_show2.reset_index(inplace = True)  
    # food_similar_show.drop(["index"], axis=1, inplace=True)
    #print (food_similar_show2)
    # return (food_similar_show)



# Enter your Order Name and number of Order's to be recommended - This is based on Order1
get_recommendations("Mysore Bonda, Poha, Hotdog Sandwich", topN = 5)


# Enter your Order Name and number of Order's to be recommended - This is based on Order2
get_recommendations1("Paneer Amritsari Tikka, Butter Garlic Prawns, Chilli Panner, Apollo Fish, Veg Tempura, Fish Tacos, Tomato and Basil Bruschetta, Tuscan Lamb Shanks, Lettuce Wraps, Thai Fried Calamari With Sweet Chili", topN = 5)


# Enter your Order Name and number of Order's to be recommended - This is based on Order3
get_recommendations2("Farm Fresh Pizza, Corn Cheese Pizza", topN = 5)


import pickle

pickle.dump(food, open('order1.pkl','wb'))

food["order1"].values

food.to_dict()

pickle.dump(food.to_dict(), open('order1.pkl','wb'))

pickle.dump(cosine_sim_matrix, open('similarity.pkl','wb'))

food.iloc[15].order1






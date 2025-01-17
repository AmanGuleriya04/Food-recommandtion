# Food-recommandtion
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "Item": ["coffee", "tea", "burger", "pizza", "sandwich", "croissant", "fries", "garlic bread"],
    "Pairings": [
        "croissant donut chocolate cake",
        "biscuits sandwich scones",
        "fries coke onion rings",
        "garlic bread cold drink chicken wings",
        "tea biscuits salad",
        "coffee butter jam",
        "burger coke ketchup",
        "pizza cheese dip herbs"
    ]
}

df = pd.DataFrame(data)

df['Combined'] = df['Item'] + " " + df['Pairings']

vectorizer = CountVectorizer()
vector_matrix = vectorizer.fit_transform(df['Combined'])

cosine_sim = cosine_similarity(vector_matrix, vector_matrix)

def get_recommendations(item):
    if item not in df['Item'].values:
        return None
    item_idx = df[df['Item'] == item].index[0]
    similarity_scores = list(enumerate(cosine_sim[item_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:4]
    recommended_items = [df['Item'][i[0]] for i in sorted_scores]
    return recommended_items

print("Welcome to the Advanced Food Recommendation System!")

user_item = input("Enter a food or drink item: ").lower()

suggested_items = get_recommendations(user_item)

if suggested_items:
    print(f"\nYou ordered: {user_item.capitalize()}")
    print("Here are some recommended items to pair with it:")
    for i, suggestion in enumerate(suggested_items, 1):
        print(f"{i}. {suggestion}")
else:
    print(f"\nSorry, we don't have recommendations for '{user_item}'.")
    print("Try ordering something classic like fries or a cold drink!")

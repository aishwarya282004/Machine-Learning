from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample dataset: User, Item, Rating
data = [
    ('A', 'Item1', 5),
    ('A', 'Item2', 3),
    ('A', 'Item3', 4),
    ('B', 'Item1', 4),
    ('B', 'Item2', 2),
    ('B', 'Item3', 5),
    ('C', 'Item1', 3),
    ('C', 'Item2', 5),
    ('C', 'Item3', 3),
]

# Convert the data into a format suitable for the surprise library
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(pd.DataFrame(data, columns=['user', 'item', 'rating']), reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.25)

# Use the Singular Value Decomposition (SVD) algorithm for collaborative filtering
model = SVD()

# Train the model
model.fit(trainset)

# Evaluate the model on the testset
predictions = model.test(testset)

# Calculate RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Predict the rating for a new user-item pair (for instance, User 'A' and 'Item3')
pred = model.predict('A', 'Item3')
print(f"Predicted rating for user 'A' on 'Item3': {pred.est}")

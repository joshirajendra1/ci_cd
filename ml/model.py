"""
Your module description
"""
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader

# import model and data module for ML
from src.regression import Regression
from src.input_data import InputData

device = "cpu"
feature_size = 17
model = Regression(feature_size)
model.to(device)

encoder_path = "model/OneHotEncoder.pickle"
with open(encoder_path, "rb") as f:
    vectorizer = pickle.load(f)


model_path = "model/model3.pth"
with open(model_path, "rb") as f:
    model.load_state_dict(torch.load(f))
    model.eval()

# model.load_state_dict(torch.load('model3.pth'))
# model.eval()


def PredictScore(user_query):
    # get users query using request
    user_query = list(user_query.values())

    # get users query using request
    user_query = list(user_query.values())

    print("user_query:  ", user_query)

    # vectorize the user's query to be used as feature for prediction
    new_data = pd.DataFrame(user_query).T
    sample_target = pd.DataFrame([78])

    print("user_query:  ", user_query)

    # vectorize the user's query to be used as feature for prediction
    new_data = pd.DataFrame(user_query).T
    sample_target = pd.DataFrame([78])

    encoded_vector = vectorizer.transform(new_data)
    encoded_vector = pd.DataFrame(encoded_vector.toarray())

    print(encoded_vector)

    sample_dataset = InputData(
        torch.from_numpy(encoded_vector.values).float(),
        torch.from_numpy(sample_target.values).float(),
    )
    sample_data_loader = DataLoader(dataset=sample_dataset, batch_size=1)

    # Predict with model
    output = []
    with torch.no_grad():
        for X_batch, _ in sample_data_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            output.append(y_test_pred.cpu().numpy()[0][0] * 100)
    print(str(output[0]))
    return str(output[0])

from model import NLPModel
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_model():
    model = NLPModel()

    PATH_TO_TRAIN_DATA = 'data/train.csv'
    PATH_TO_TEST_DATA = 'data/test.csv'
    train_df = pd.read_csv(PATH_TO_TRAIN_DATA)
    test_df = pd.read_csv(PATH_TO_TEST_DATA)
    
    train_comment = train_df['comment']
    test_comment = test_df['comment']
    train_true = train_df['label']
    test_true = test_df['label']
    
    model.vectorizer_fit(train_comment)
    print('Vectorizer fit complete')

    train_transformed = model.vectorizer_transform(train_comment)
    test_transformed = model.vectorizer_transform(test_comment)
    print('Vectorizer transform completed')

    model.train(train_transformed, train_true)
    print('Model training completed')
    
    predicted = model.predict(test_transformed)
    print('Model testing completed')
    accuracy = model.eval(test_true,predicted)
    print('The accuracy of the model is',accuracy)
    
    model.pickle_clf()
    model.pickle_vectorizer()
    

if __name__ == "__main__":
    build_model()


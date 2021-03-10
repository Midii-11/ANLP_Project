import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def encode(df: pd.DataFrame, column_y: str, column_x: str):
    """
    This function prepares the x and y for training.
    It returns:
        - a numpy array containing the OneHot encoded 'y' categories
        - Pandas series containing the preprocessed text of each 'x'


    :param df:              Input the Dataframe of interest
    :param column_y:        Name of the column of interest for the 'y' label
    :param column_x         Name of the column of interest for the 'x' label
    :return:                Returns a
    """
    y = df.loc[:, column_y]
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    x = df.loc[:, column_x]
    return y, x


def bag_of_words(x_train: [], x_test: []) -> []:
    """
    This function encodes the words of each elements of x_train and x_test to Bag Of Words (BOW).

    :param x_train:             OneHot encoded 'y' categories
    :param x_test:              Pandas series containing the preprocessed text of each 'x'
    :return:                    Returns the training and testing sets.
    """
    #TODO: Could implement 2 dataframes holding the x_text_bow_train of 'history_text' and 'powers_text'. One for
    #       x_text_bow_train and one for x_text_bow_test.
    # This would enable us to use multiple-inputs models.
    # eg: Use 'history_text' AND 'powers_text' to predict the creator.

    # defining the bag-of-words transformer on the text-processed corpus
    bow_transformer = CountVectorizer(analyzer='word').fit(x_train)
    # transforming into Bag-of-Words and hence textual data to numeric..
    x_text_bow_train = bow_transformer.transform(x_train)
    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_test = bow_transformer.transform(x_test)

    return x_text_bow_train, text_bow_test


def split_data(y: [], x: pd.Series):
    """
    This function split the data into training and testing sets.

    :param y:               OneHot encoded 'y' categories
    :param x:               Pandas series containing the preprocessed text of each 'x'
    :return:                Returns the training and testing sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
    return x_train, x_test, y_train, y_test


def linear_regression(x_text_bow_train, y_train, SAVE):
    """
    This function trains the model using linear regression

    :param x_text_bow_train:        x_train data to use
    :param y_train:                 y_train data to use
    :return:                        Returns a fitted model
    """
    # instantiating the model with simple Logistic Regression..
    model = LogisticRegression()
    # training the model...
    model = model.fit(x_text_bow_train, y_train)

    # save the model to disk
    if SAVE is True:
        filename = 'LinearRegression.sav'
        joblib.dump(model, 'models/' + filename)
    return model


def assess_model(model, x_train, x_test, y_train, y_test, DISPLAY):
    """
    This function asses the input model

    :param model:                   Input model to be assessed
    :param x_train:                 x_train data to use
    :param x_test                   x_test data to use
    :param y_train                  y_train data to use
    :param y_test                   y_test data to use
    :param DISPLAY:                 True if graphs are expected
    """
    print(model.score(x_train, y_train))
    print(model.score(x_test, y_test))

    target_names = ['Marvel', 'DC', 'Other']

    # getting the predictions of the Validation Set...
    predictions = model.predict(x_test)
    # getting the Precision, Recall, F1-Score
    print(classification_report(y_test, predictions, target_names=target_names))

    # displays a confusion matrix of the model performance on the test set
    if DISPLAY is True:
        confusion = confusion_matrix(y_test, predictions)
        print('Confusion Matrix')
        print(confusion)
        fig, ax = plot.subplots(figsize=(7.5, 7.5))
        ax.matshow(confusion, cmap=plot.cm.winter, alpha=0.3)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center', size='xx-large')

        plot.xlabel('Predictions', fontsize=18)
        plot.ylabel('Actuals', fontsize=18)
        plot.title('Confusion Matrix', fontsize=18)
        plot.show()


if __name__ == '__main__':
    column_head_y = 'creator'
    column_head_x = 'history_text'

    # DISPLAY = True
    DISPLAY = False
    SAVE = True
    # SAVE = False

    # Load the Dataset and Drop the np.nan values of the table
    df = pd.read_csv('datasets/Preprocessed.csv')
    df_interest = df.dropna(subset=[column_head_y, column_head_x])

    # TODO: Try multiple-inputs models (--> multiple linear regression)
    # Get the x and y for the model
    y, x = encode(df_interest, column_head_y, column_head_x)
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = split_data(y, x)
    # Encode X with Bag Of Words (BOW)
    x_text_bow_train, x_text_bow_test = bag_of_words(x_train, x_test)

    # Fit a Linear Regression model
    linear_regression_model = linear_regression(x_text_bow_train, y_train, SAVE)
    # TODO: Test other models and Test TFIDF instead of BOW
    # Assess the trained model
    assess_model(linear_regression_model, x_text_bow_train, x_text_bow_test, y_train, y_test, DISPLAY)




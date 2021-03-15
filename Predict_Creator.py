import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec


def encode(df: pd.DataFrame, column_y: str, column_x: str):
    """
    This function prepares the x and y for training.

    :param df:              Input the Dataframe of interest
    :param column_y:        Name of the column of interest for the 'y' label
    :param column_x         Name of the column of interest for the 'x' label
    :return:                Returns     - x_init as a series containing the text of each cells
                                        - y_init as an array containing the OneHot representation of the labels
    """
    y_init = df.loc[:, column_y]
    labelencoder = LabelEncoder()
    y_init = labelencoder.fit_transform(y_init)
    x_init = df.loc[:, column_x]

    return x_init, y_init


def split_data(x_init, y_init) -> []:
    """
    This function split the data into training and testing sets.

    :param x_init:              Pandas series containing the text of each cells
    :param y_init:              Array containing the OneHot representation of the labels
    :return:                    Returns the training and testing sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x_init, y_init, test_size=0.35, random_state=seed)
    return x_train, x_test, y_train, y_test


def bag_of_words(x_train, x_test):
    """
    This function encodes the words of each elements of x_train and x_test to Bag Of Words (BOW).

    :param x_train:             Series containing the text of each cell of the training set
    :param x_test:              Series containing the text of each cell of the testing set
    :return:                    Returns the X training and testing sets processed by BOW (txt --> num)
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
    x_text_bow_test = bow_transformer.transform(x_test)

    return x_text_bow_train, x_text_bow_test


def tf_idf(x_train, x_test):
    """
        This function encodes the words of each elements of x_train and x_test to TFIDF.

        :param x_train:             Series containing the text of each cell of the training set
        :param x_test:              Series containing the text of each cell of the testing set
        :return:                    Returns the X training and testing sets processed by TFIDF (txt --> num)
        """
    #TODO: check if this uses a weighted sum for tfidf calculation
    tfidf = TfidfVectorizer(stop_words='english', analyzer='word', strip_accents='unicode', sublinear_tf=True,
                            token_pattern=r'\w{1,}', max_features=5000, ngram_range=(1, 2))
    tfidf.fit(x_train)
    tfidf.fit(x_test)
    x_train_tfidf = tfidf.transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    return x_train_tfidf, x_test_tfidf


def linear_regression(x_train, x_test, y_train, y_test, SAVE, model_name, column_head_x):
    """
    This function trains the model using linear regression

    :param x_train:             X_train values to use to train the model --> x_train_[bow/tfidf]
    :param x_test:              X_train values to use to train the model --> x_test_[bow/tfidf]
    :param y_train:             y_train values to use to train the model --> (OneHot Encoded)
    :param y_test:              y_test values to use to train the model --> (OneHot Encoded)
    :param SAVE:                Save the model [Boolean]
    :param model_name:          Model name

    :return:                    Returns the X training and testing sets processed by BOW (txt --> num)
    """

    # instantiating the model with simple Logistic Regression..
    model = LinearRegression()
    # training the model...
    model = model.fit(x_train, y_train)

    if SAVE is True:
        print(model_name, ":", file=open("Terminal_Output.txt", "a"))
        print(model.score(x_train, y_train), file=open("Terminal_Output.txt", "a"))
        print(model.score(x_test, y_test), file=open("Terminal_Output.txt", "a"))
        print('\n', file=open("Terminal_Output.txt", "a"))
    else:
        print(model_name, ":")
        print(model.score(x_train, y_train))
        print(model.score(x_test, y_test))
        print('\n')
    # TODO: Graph --> line best fit

    # save the model to disk
    if SAVE is True:
        joblib.dump(model, 'models/' + model_name + '_' + column_head_x + '.sav')
    return model


def logistic_regression(x_train, y_train, SAVE, model_name, column_head_x):
    """
    This function trains the model using linear regression

    :param x_train:        x_train data to use
    :param y_train:                 y_train data to use
    :return:                        Returns a fitted model
    """
    # instantiating the model with simple Logistic Regression..
    model = LogisticRegression()
    # training the model...
    model = model.fit(x_train, y_train)

    # save the model to disk
    if SAVE is True:
        joblib.dump(model, 'models/' + model_name + '_' + column_head_x + '.sav')
    return model


def naive_bayes(x_train, y_train, SAVE, model_name, column_head_x):
    # instantiating the model with simple Logistic Regression..
    model = MultinomialNB()
    # training the model...
    model = model.fit(x_train, y_train)

    # save the model to disk
    if SAVE is True:
        joblib.dump(model, 'models/' + model_name + '_' + column_head_x + '.sav')
    return model


def svm_classifier(x_train, y_train, SAVE, model_name, column_head_x):
    # instantiating the model with simple Logistic Regression..
    model = SVC()
    # training the model...
    model = model.fit(x_train, y_train)

    # save the model to disk
    if SAVE is True:
        joblib.dump(model, 'models/' + model_name + '_' + column_head_x + '.sav')
    return model


def random_forest_bow(x_train, y_train, SAVE, model_name, column_head_x):
    # instantiating the model with simple Logistic Regression..
    model = RandomForestClassifier()
    # training the model...
    model = model.fit(x_train, y_train)

    # save the model to disk
    if SAVE is True:
        joblib.dump(model, 'models/' + model_name + '_' + column_head_x + '.sav')
    return model


def assess_model(model, x_train, x_test, y_train, y_test, DISPLAY, model_name, concat_status, column_head_x):
    """
    This function asses the input model

    :param model:                   Input model to be assessed
    :param x_train:                 x_train data to use
    :param x_test                   x_test data to use
    :param y_train                  y_train data to use
    :param y_test                   y_test data to use
    :param DISPLAY:                 True if graphs are expected
    """

    if SAVE is True:
        print(model_name, ' score:', file=open("Terminal_Output.txt", "a"))
        print(model.score(x_train, y_train), file=open("Terminal_Output.txt", "a"))
        print(model.score(x_test, y_test), file=open("Terminal_Output.txt", "a"))
    else:
        print(model_name, ' score:')
        print(model.score(x_train, y_train))
        print(model.score(x_test, y_test))

    target_names = ['Marvel', 'DC', 'Other']

    # getting the predictions of the Validation Set...
    predictions = model.predict(x_test)
    # getting the Precision, Recall, F1-Score
    if SAVE is True:
       print(classification_report(y_test, predictions, target_names=target_names), file=open("Terminal_Output.txt", "a"))
    else:
        print(classification_report(y_test, predictions, target_names=target_names))

    # displays a confusion matrix of the model performance on the test set
    if DISPLAY is True:
        confusion = confusion_matrix(y_test, predictions)
        if SAVE is True:
            print(model_name, ' Confusion Matrix', file=open("Terminal_Output.txt", "a"))
            print(confusion, file=open("Terminal_Output.txt", "a"))
            print('\n', file=open("Terminal_Output.txt", "a"))
        else:
            print(model_name, ' Confusion Matrix')
            print(confusion)
            print('\n')
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(confusion, cmap=plt.cm.winter, alpha=0.3)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                ax.text(x=j, y=i, s=confusion[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        if concat_status is True:
            title_name = model_name + '\nCreator classification from History and Power text'
        else:
            title_name = model_name + '\nCreator classification from only History'

        plt.title(title_name, fontsize=18)
        plt.tight_layout()
        if SAVE is True:
            plt.savefig("Figures/" + model_name + '_' + column_head_x)
        plt.show()

if __name__ == '__main__':

    column_head_y = 'creator'
    # column_head_x = 'history_text'
    column_head_x = 'history_power_text'

    if column_head_x == 'history_power_text':
        concat_status = True
    else:
        concat_status = False
    seed = 42

    DISPLAY = True
    # DISPLAY = False
    # SAVE = True
    SAVE = False

    # Load the Dataset and Drop the np.nan values of the table
    df = pd.read_csv('datasets/Preprocessed.csv')

    df.loc[:, column_head_x] = df.loc[:, 'history_text'].astype(str) + df.loc[:, 'powers_text'].astype(str)



    df_interest = df.dropna(subset=[column_head_y, column_head_x])


    # TODO: Try multiple-inputs models (--> multiple linear regression)

    # Get initial x and y ( X=text, Y=OneHotEncoded labels )
    x_init, y_init = encode(df_interest, column_head_y, column_head_x)
    x_train, x_test, y_train, y_test = split_data(x_init, y_init)   # Split the data into training and testing sets

    x_train_bow, x_test_bow = bag_of_words(x_train, x_test)         # Encode X with Bag Of Words (BOW)
    x_train_tfidf, x_test_tfidf = tf_idf(x_train, x_test)           # Encode X with TFIDF

    #
    # TODO: check if linear regression even makes sense for the task
    # Linear Regression model using BOW and TFIDF
    linear_regression_model_bow = linear_regression(x_train_bow, x_test_bow, y_train, y_test, SAVE, 'Linear_Regression_BOW', column_head_x)
    linear_regression_model_tfidf = linear_regression(x_train_tfidf, x_test_tfidf, y_train, y_test, SAVE, 'Linear_Regression_TFIDF', column_head_x)

    # Logistic Regression model using BOW and TFIDF
    logistic_regression_model_bow = logistic_regression(x_train_bow, y_train, SAVE, 'Logistic_Regression_BOW', column_head_x)
    logistic_regression_model_tfidf = logistic_regression(x_train_tfidf, y_train, SAVE, 'Logistic_Regression_TFIDF', column_head_x)

    # Naive Bayes model using BOW and TFIDF
    naive_bayes_model_bow = naive_bayes(x_train_bow, y_train, SAVE, 'Naive_Bayes_BOW', column_head_x)
    naive_bayes_model_tfidf = naive_bayes(x_train_tfidf, y_train, SAVE, 'Naive_Bayes_TFIDF', column_head_x)

    # SVM model using BOW and TFIDF
    svm_model_bow = svm_classifier(x_train_bow, y_train, SAVE, 'svm_BOW', column_head_x)
    svm_model_TFIDF = svm_classifier(x_train_tfidf, y_train, SAVE, 'svm_TFIDF', column_head_x)

    # Random Forest model using BOW and TFIDF
    random_forest_model_bow = svm_classifier(x_train_bow, y_train, SAVE, 'Random_Forest_BOW', column_head_x)
    random_forest_model_tfidf = svm_classifier(x_train_tfidf, y_train, SAVE, 'Random_Forest_TFIDF', column_head_x)


    # Assess the trained models
    assess_model(logistic_regression_model_bow, x_train_bow, x_test_bow, y_train, y_test, DISPLAY,
                 'Logistic Regression BOW', concat_status, column_head_x)
    assess_model(logistic_regression_model_tfidf, x_train_tfidf, x_test_tfidf, y_train, y_test, DISPLAY,
                 'Logistic Regression TFIDF', concat_status, column_head_x)
    assess_model(naive_bayes_model_bow, x_train_bow, x_test_bow, y_train, y_test, DISPLAY,
                 'Naive Bayes BOW', concat_status, column_head_x)             # The IDE freaks out but it works
    assess_model(naive_bayes_model_tfidf, x_train_tfidf, x_test_tfidf, y_train, y_test, DISPLAY,
                 'Naive Bayes TFIDF', concat_status, column_head_x)           # The IDE freaks out but it works
    # svm_model_bow throw a warning due to the fact that one of the label is never predicted by the model --> bad model
    assess_model(svm_model_bow, x_train_bow, x_test_bow, y_train, y_test, DISPLAY,
                 'SVM BOW', concat_status, column_head_x)                     # The IDE freaks out but it works
    assess_model(svm_model_TFIDF, x_train_tfidf, x_test_tfidf, y_train, y_test, DISPLAY,
                 'SVM TFIDF', concat_status, column_head_x)                   # The IDE freaks out but it works
    # random_forest_model_bow throw a warning due to one of the label is never predicted by the model --> bad model
    assess_model(random_forest_model_bow, x_train_bow, x_test_bow, y_train, y_test, DISPLAY,
                 'Random Forest BOW', concat_status, column_head_x)  # The IDE freaks out but it works
    assess_model(random_forest_model_tfidf, x_train_tfidf, x_test_tfidf, y_train, y_test, DISPLAY,
                 'Random Forest TFIDF', concat_status, column_head_x)  # The IDE freaks out but it works








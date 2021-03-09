import re
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from nltk.corpus import stopwords
import spacy

from tqdm import tqdm
# nltk.download('punkt')

def goal_overall_score(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    This function preprocesses the "overall score" column to only contain floats. It replaces the following symbols
    by float values: '∞' / '-'.

    :param df:              Input the Dataframe of interest
    :param column_name:     Name of the column of interest
    :return:                Returns a Dataframe containing the preprocessed column
    """

    # Try to convert Overall_score from string to int, else convert symbols to values
    df.loc[df.loc[:, column_name] == '∞', column_name] = 250.
    df.loc[df.loc[:, column_name] == '-', column_name] = np.nan
    df.loc[:, column_name] = df.loc[:, column_name].astype(float)

    print('\n Preprocessed shape:\t', df.shape)

    # Show Histogram of the Overall_scores
    if DISPLAY is True:
        plot.hist(df.overall_score, bins=250)
        plot.title("Histogram: Overall score")
        plot.xticks(np.arange(0, 250, 25))
        plot.grid()
        plot.tight_layout()
        plot.show()

    # Prints the different dtypes in the column of interest
    verification(df, column_name)
    # Temporarily save the dataset for analysis
    save_temp_df(df)

    return df


def goal_creators(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
        This function preprocesses the "creator" column. It converts the smaller creators to "Other"

    :param df:              Input the Dataframe of interest
    :param column_name:     Name of the column of interest
    :return:                Returns a Dataframe containing the preprocessed column
    """

    # Displays a pie chart representation of the creators
    if DISPLAY is True:
        creators = df.creator.value_counts().to_frame().T  # todo: double check if this is clean code
        # print(df.creator.value_counts().to_frame().T)
        plot.subplots(figsize=(10, 8))
        patches, texts = plot.pie(x=creators.values[0], startangle=90)
        plot.legend(patches, list(creators), loc="best", bbox_to_anchor=(1, 1))
        plot.title('Pie chart: Creators')
        plot.show()

    # Replace smaller creators by 'other' -> 3 categories and leaves nan as is
    for index in df.index:
        if df.iloc[index, 5] != 'Marvel Comics' and df.iloc[index, 5] != 'DC Comics' and isinstance(
                df.iloc[index, 5], float) is False:
            df.iloc[index, 5] = 'Other'

    # Displays a pie chart representation of the creators (Marvel / DC / Other)
    if DISPLAY is True:
        creators = df.creator.value_counts().to_frame().T
        plot.subplots(figsize=(10, 8))
        patches, texts = plot.pie(x=creators.values[0], startangle=90)
        plot.legend(patches, list(creators), loc="best", bbox_to_anchor=(1, 1))
        plot.title('Pie chart: Creators (Grouped)')
        plot.show()

    # Prints the different dtypes in the column of interest
    verification(df, column_name)
    # Temporarily save the dataset for analysis
    save_temp_df(df)

    return df


def goal_alignment(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    This function preprocesses the "alignment" column. It does not change the column since it is already correctly 
    formated. 

    :param df:              Input the Dataframe of interest
    :param column_name:     Name of the column of interest
    :return:                Returns a Dataframe containing the preprocessed column
    """

    if DISPLAY is True:
        alignments = df_interest.alignment.value_counts().to_frame().T

        # Plot alignment proportion Pie Chart
        plot.subplot(2, 1, 1)
        patches, texts = plot.pie(x=alignments.values[0], startangle=90)
        plot.legend(patches, list(alignments), loc="best", bbox_to_anchor=(1, 1))
        plot.title('Pie chart: Alignment proportion')

        # Plot alignment proportion Bar Chart
        plot.subplot(2, 1, 2)
        plot.bar(list(alignments), alignments.values[0])
        plot.title("Bar chart: Alignment proportion")
        plot.grid()
        plot.tight_layout()
        plot.show()

    # Prints the different dtypes in the column of interest
    verification(df, column_name)
    # Temporarily save the dataset for analysis
    save_temp_df(df)

    return df


def goal_superpowers(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    # TODO: Finish documentation
    This function preprocesses the "superpowers" column. It assigns a np.nan value to empty inputs ('[]').

    :param df:              Input the Dataframe of interest
    :param column_name:     Name of the column of interest
    :return:                Returns a Dataframe containing the preprocessed column
    """

    # TODO: Plot superpower repartition
    total = []
    # Transform empty superpower lists to NaN, otherwise transform superpowers from str to array.
    for index in df.index:
        if df.loc[index, column_name] == '[]':
            df.loc[index, column_name] = np.nan
        # TODO: check if finished
        # else:
        #     separated = np.array(df.loc[index, column_name].split(sep=','))
        #     for i in range(len(separated)):
        #         separated[i] = re.sub(r'[^\w\s]', '', str(separated[i]))
        #     df.loc[index, column_name] = str(separated)
        #     print(df.loc[index, column_name])

    # Prints the different dtypes in the column of interest
    verification(df, column_name)
    # Temporarily save the dataset for analysis
    # save_temp_df(df)
    return df


def goal_text(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    This function preprocesses the "history_text" column. It:
    - Transforms the text to lower cases
    - Removes the punctuation
    - Replace numbers by '#'
    - Removes stopwords using NLTK corpus
    - Lemmatize the remaining text using Spacy's en_core_web_sm

    /!\ To process the whole dataset, it took 06:45 min on an Intel i7-9700k
    Python used a single core --> Multi-thread possible ?

    :param df:              Input the Dataframe of interest
    :param column_name:     Name of the column of interest
    :return:                Returns a Dataframe containing the preprocessed column
    """

    for index in tqdm(df.index):
        text = df.loc[index, column_name]               # Get the original text from individual cell
        text = text.lower()                             # Converts text into lower cases
        text = text.translate(str.maketrans('', '', string.punctuation))    # Most efficient way to remove punctuation

        text = re.sub('[0-9]{5,}', '#####', text)       # Replace numerical values by '#'
        text = re.sub('[0-9]{4}', '####', text)
        text = re.sub('[0-9]{3}', '###', text)
        text = re.sub('[0-9]{2}', '##', text)
        text = re.sub('[0-9]{1}', '#', text)

        # Remove stopwords from NLTK corpus
        cached_stop_words = stopwords.words("english")
        text = " ".join([word for word in text.split() if word not in cached_stop_words])

        # Lemmatize the remaining words
        nlp = spacy.load('en_core_web_sm')              # run " python -m spacy download en_core_web_sm " in terminal
        doc = nlp(text)
        tokens = []
        for token in doc:
            tokens.append(token)
        lemmatized_sentence = " ".join([token.lemma_ for token in doc])

        # Append the lemmatized sentences to the dataframe
        df.loc[index, column_name] = lemmatized_sentence

    # Prints the different dtypes in the column of interest
    verification(df, column_name)
    # Temporarily save the dataset for analysis
    # save_temp_df(df)

    return df


def verification(df: pd.DataFrame, column_name: str) -> None:
    """
    Prints the different data types present in the column of interest.
    """
    types = []
    for index in df.index:
        if type(df.loc[index, column_name]) not in types:
            types.append(type(df.loc[index, column_name]))
    print(column_name, ':\t', types)


def save_temp_df(df: pd.DataFrame) -> None:
    df.to_csv('datasets/temp.csv', index=False)


if __name__ == '__main__':

    # Displays graphs
    # DISPLAY = True
    DISPLAY = False

    # Saves DF
    SAVE = True
    # SAVE = False

    # Load data into a pandas dataframe
    df_full = pd.read_csv("datasets/superheroes_nlp_dataset.csv")
    df_interest = df_full.loc[:, ("name", "overall_score", "history_text", "powers_text",
                                  "superpowers", "creator", "alignment")]

    # Get the number of null entries in the interesting data
    print('\n Initial shape:\t', df_interest.shape)
    print(df_interest.isnull().sum())

    # Remove rows containing empty values in main categories
    df_interest.dropna(subset=["history_text", "powers_text"], inplace=True)
    df_interest.reset_index(drop=True, inplace=True)

    # -----------------------------------
    # Calls functions
    # -----------------------------------
    # Prepares the dataset for Overall_score prediction from history and power texts
    df_interest = goal_overall_score(df_interest, 'overall_score')
    # Prepares the dataset for Creator prediction from history and power texts
    df_interest = goal_creators(df_interest, 'creator')
    # Prepares the dataset for Alignment prediction from history and power texts
    df_interest = goal_alignment(df_interest, 'alignment')
    # Prepares the dataset for Superpowers prediction from history and power texts
    df_interest = goal_superpowers(df_interest, 'superpowers')
    # Prepares the history texts
    df_interest = goal_text(df_interest, 'history_text')
    # Prepares the history texts
    df_interest = goal_text(df_interest, 'powers_text')


    # Save the preprocessed dataset
    if SAVE is True:
        df_interest.to_csv("datasets/Preprocessed.csv", index=False)
        print('\n Dataset SAVED.')

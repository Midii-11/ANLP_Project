import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import nltk
from nltk.tokenize import RegexpTokenizer

# nltk.download('punkt')


def goal_Overall_score():
    # Try to convert Overall_score from string to int, else convert symbols to values
    for index in df_interest.index:
        try:
            df_interest.iloc[index, 1] = int(df_interest.iloc[index, 1])
        except:
            if df_interest.iloc[index, 1] == '∞':
                df_interest.iloc[index, 1] = 250
            else:
                df_interest.iloc[index, 1] = np.nan
        finally:
            # print(index, '\t', df_interest.iloc[index, 1])
            pass
    print('\n Preprocessed shape:\t', df_interest.shape)

    # Show Histogram of the Overall_scores
    if DISPLAY is True:
        plot.hist(df_interest.overall_score, bins=250)
        plot.title("Histogram: Overall score")
        plot.xticks(np.arange(0, 250, 25))
        plot.grid()
        plot.tight_layout()
        plot.show()


def goalCreators():
    # Displays a pie chart representation of the creators
    if DISPLAY is True:
        creators = df_interest.creator.value_counts().to_frame().T
        # print(df_interest.creator.value_counts().to_frame().T)
        plot.subplots(figsize=(10, 8))
        patches, texts = plot.pie(x=creators.values[0], startangle=90)
        plot.legend(patches, list(creators), loc="best", bbox_to_anchor=(1, 1))
        plot.title('Pie chart: Creators')
        plot.show()

    # Replace smaller creators by 'other' -> 3 categories and leaves nan as is
    for index in df_interest.index:
        if df_interest.iloc[index, 5] != 'Marvel Comics' and df_interest.iloc[index, 5] != 'DC Comics' and isinstance(df_interest.iloc[index, 5], float) is False:
            df_interest.iloc[index, 5] = 'Other'

    # Displays a pie chart representation of the creators (Marvel / DC / Other)
    if DISPLAY is True:
        creators = df_interest.creator.value_counts().to_frame().T
        # print(df_interest.creator.value_counts().to_frame().T)
        plot.subplots(figsize=(10, 8))
        patches, texts = plot.pie(x=creators.values[0], startangle=90)
        plot.legend(patches, list(creators), loc="best", bbox_to_anchor=(1, 1))
        plot.title('Pie chart: Creators (Grouped)')
        plot.show()

    # print(df_interest.creator)


def goalAlignment():

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



def goalSuperpowers():

    total = []
    # Transform empty superpower lists to NaN, otherwise transform superpowers from str to array.
    for index in df_interest.index:
        if df_interest.iloc[index, 4] == '[]':
            df_interest.iloc[index, 4] = np.nan
        else:


            tokenizer = RegexpTokenizer(r'\w+')
            tok = tokenizer.tokenize(df_interest.iloc[index, 4])
            print(df_interest.iloc[index, 4] )
            print(tok)


if __name__ == '__main__':

    # Displays graphs
    # DISPLAY = True
    DISPLAY = False

    # Saves DF
    # SAVE = True
    SAVE = False

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
    # Create the dataset for Overall_score prediction from history and power texts
    goal_Overall_score()
    # Create the dataset for Creator prediction from history and power texts
    goalCreators()
    # Create the dataset for Alignment prediction from history and power texts
    goalAlignment()
    # Create the dataset for Superpowers prediction from history and power texts
    # goalSuperpowers()

    # Save the preprocessed dataset
    if SAVE is True:
        df_interest.to_csv("datasets/Preprocessed.csv", index=False)
        print('\n Dataset SAVED.')

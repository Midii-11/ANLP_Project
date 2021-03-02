import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


def goal_Overall_score(df_interest, DISPLAY):
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
    return df_interest


def goalCreators(df_interest):
    # Displays a pie chart representation of the creators
    if DISPLAY is True:
        creators = df_interest.creator.value_counts().to_frame().T
        # print(df_interest.creator.value_counts().to_frame().T)
        plot.subplots(figsize=(10, 8))
        patches, texts = plot.pie(x=creators.values[0], startangle=90)
        plot.legend(patches, list(creators), loc="best", bbox_to_anchor=(1, 1))
        plot.title('Pie chart: Creators')
        plot.show()

    # Replace smaller creators by 'other' -> 3 categories
    for index in df_interest.index:
        # TODO: atm, nan values are replaced by 'other' --> should keep as np.nan ?
        if df_interest.iloc[index, 5] != 'Marvel Comics' and df_interest.iloc[index, 5] != 'DC Comics' and df_interest.iloc[index, 5] != np.nan:
            df_interest.iloc[index, 5] = 'Other'

    # Displays a pie chart representation of the creators (Marvel / DC / Other)
    if DISPLAY is True:
        creators = df_interest.creator.value_counts().to_frame().T
        # print(df_interest.creator.value_counts().to_frame().T)
        plot.subplots(figsize=(10, 8))
        patches, texts = plot.pie(x=creators.values[0], startangle=90)
        plot.legend(patches, list(creators), loc="best", bbox_to_anchor=(1, 1))
        plot.title('Pie chart: Creators (Gathered)')
        plot.show()

    print(df_interest.creator)
    return df_interest


def goalAlignment(df_interest):


    return df_interest



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
    # Create the dataset for Overall_score prediction from history and power texts
    df_interest = goal_Overall_score(df_interest, DISPLAY)
    # Create the dataset for Creator prediction from history and power texts
    df_interest = goalCreators(df_interest)
    df_interest = goalAlignment(df_interest)

    # Save the preprocessed dataset
    if SAVE is True:
        df_interest.to_csv("datasets/Preprocessed.csv", index=False)
        print('\n Dataset SAVED.')

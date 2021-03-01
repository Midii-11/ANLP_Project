import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


def goal_Overall_score(df_interest, DISPLAY):
    for index in df_interest.index:
        try:
            df_interest.iloc[index, 1] = int(df_interest.iloc[index, 1])
        except:
            if df_interest.iloc[index, 1] == '∞':
                df_interest.iloc[index, 1] = 250
            else:
                df_interest.iloc[index, 1] = np.nan
        finally:
            print(index, '\t', df_interest.iloc[index, 1])
    print(df_interest.shape)


    if DISPLAY is True:
        plot.hist(df_interest.overall_score, bins=250)
        plot.title("Histogram: Overall score")
        plot.xticks(np.arange(0, 250, 25))
        plot.grid()
        plot.tight_layout()
        plot.show()
    return df_interest





if __name__ == '__main__':

    # Displays graphs
    DISPLAY = True
    # Saves DF
    SAVE = True

    # Load data into a pandas dataframe
    df_full = pd.read_csv("datasets/superheroes_nlp_dataset.csv")
    df_interest = df_full.loc[:, ("name", "overall_score", "history_text", "powers_text",
                                  "superpowers", "creator", "alignment")]

    # Get the number of null entries in the interesting data
    print(df_interest.isnull().sum())
    print(df_interest.shape)

    # Remove rows containing empty values in main categories
    df_interest.dropna(subset=["history_text", "powers_text"], inplace=True)
    df_interest.reset_index(drop=True, inplace=True)

    # Create the dataset for Overall_score prediction from history and power txts
    df_interest = goal_Overall_score(df_interest, DISPLAY)

    if SAVE is True:
        df_interest.to_csv("datasets/Preprocessed.csv")

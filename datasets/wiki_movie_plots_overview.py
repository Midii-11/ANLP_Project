import pandas as pd

movie_df = pd.read_csv('datasets/wiki_movie_plots_deduped.csv')
print(movie_df.loc[:, 'Origin/Ethnicity'].unique())

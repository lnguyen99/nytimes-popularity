import pandas as pd
import sys
from nytimes_scraper.nyt_api import NytApi
from nytimes_scraper.comments import fetch_comments, comments_to_df

## Run the command as follow to scrape the number of comments per article 
## python3 scrape-comment.py <year> <month> <api_key>

## Example: 
## python3 scrape-comment.py 2020 12 svVK2Twxudkfd3IxQozMz3NyyjvsVNmZ

## Articles of the specified month must be scraped and stored at the directory articles_by_month/
## Updated dataset (articles + n_comments) of the specified month
## is stored at the directory updated_articles_by_month/

year = sys.argv[1]
month = sys.argv[2]
api_key = sys.argv[3]

name = './articles_by_month/article_' + year + '_' + month + '.csv'
updated_name = './updated_articles_by_month/updated_article_' + year + '_' + month + '.csv'

api = NytApi(api_key)

print(updated_name)
print('API KEY: ' + api_key)

# Fetch articles of a specific month
article_df = pd.read_csv(name)
article_ids_and_urls = list(article_df['web_url'].iteritems())

comments = fetch_comments(api, article_ids_and_urls)
comments_df = comments_to_df(comments)

# Get the number of comments per article
comment_series = comments_df['articleID'].value_counts().to_frame().reset_index()
comment_series.columns=['_temp_id', 'n_comments']
article_df = article_df.rename(columns={'Unnamed: 0' : '_temp_id'})

# Merge to get the updated article df 
updated_df = pd.merge(article_df, comment_series, on='_temp_id', how='outer')
updated_df = updated_df.drop(['_temp_id'], axis=1)
updated_df['n_comments'] = updated_df['n_comments'].fillna(0)

# Export to csv file 
updated_df.to_csv(updated_name)

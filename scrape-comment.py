import datetime as dt
import pandas as pd
import sys
from nytimes_scraper.nyt_api import NytApi
from nytimes_scraper.comments import fetch_comments, comments_to_df

year = sys.argv[1]
month = sys.argv[2]
api_key = sys.argv[3]

# date = dt.date(int(year), int(month), 1)
name = 'article_' + year + '_' + month + '.csv'
updated_name = 'updated_article_' + year + '_' + month + '.csv'

api = NytApi(api_key)

print(updated_name)
print('API KEY: ' + api_key)

# Fetch articles of a specific month
article_df = pd.read_csv('./articles_by_month/' + name)
article_ids_and_urls = list(article_df['web_url'].iteritems())

comments = fetch_comments(api, article_ids_and_urls)
comment_df = comments_to_df(comments)

# Get the number of comments per article
comment_series = comments_df['articleID'].value_counts().to_frame().reset_index()
comment_series.columns=['_id', 'n_comments']

# Merge to get the updated article df 
updated_df = articles_df.merge(comment_series, on=['_id'])
  
# Export to csv file 
updated_df.to_csv(updated_name)

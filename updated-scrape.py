import datetime as dt
import pandas as pd
import sys
from nytimes_scraper.nyt_api import NytApi
from nytimes_scraper import scrape_month

api_key = 'lz3ClJ5oTuckGZAhiew2IAJ6t1qTvgS3'

year = sys.argv[1]
month = sys.argv[2]

name = 'article_' + year + '_' + month + '.csv'

print(name)

# Fetch articles of a specific month
article_df, comment_df = scrape_month(api_key, date=dt.date(dt.date(int(year), int(month), 1)))

# Drop unnecessary columns 
article_df.drop(['snippet', 'print_section', 'print_page', 'source', 'multimedia',
       'uri', 'html', 'text', 'headline.kicker', 'headline.content_kicker',
       'headline.print_headline', 'headline.name', 'headline.seo',
       'headline.sub', 'byline.person', 'byline.organization'], axis=1, inplace=True)

# Get the number of comments per article
comment_series = comments_df['articleID'].value_counts().to_frame().reset_index()
comment_series.columns=['_id', 'n_comments']

# Merge to get the updated article df 
updated_df = articles_df.merge(comment_series, on=['_id'])
  
# Export to csv file 
updated_df.to_csv(name)

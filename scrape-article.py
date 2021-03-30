import datetime as dt
import pandas as pd
import sys
from nytimes_scraper.nyt_api import NytApi
from nytimes_scraper.articles import fetch_articles_by_month, articles_to_df

api = NytApi('lz3ClJ5oTuckGZAhiew2IAJ6t1qTvgS3')

year = sys.argv[1]
month = sys.argv[2]

for i in range(8):
  month += 1
  if (month > 12): 
    month = 1
    year += 1
  name = 'article_' + year + '_' + month + '.csv'
  print(name)
  # Fetch articles of a specific month
  article_df = cached(
      fetch=lambda: articles_to_df(fetch_articles_by_month(api, dt.date(int(year), int(month), 1)),
      file=out_file(date, 'articles'),
      force_fetch=force_fetch,
      store=store,
  )
  article_df.drop(['_id', 'snippet', 'print_section', 'print_page', 'source', 'multimedia',
       'uri', 'html', 'text', 'headline.kicker', 'headline.content_kicker',
       'headline.print_headline', 'headline.name', 'headline.seo',
       'headline.sub', 'byline.original', 'byline.person',
       'byline.organization', 'subsection_name'], axis=1, inplace=True
                 
  )
#   article_df.rename(columns={'headline.main': 'headline'})
  article_df.to_csv(name)
  
    
### bash code to merge 2 csv files 
### cat a.csv <(tail +2 b.csv)

import datetime as dt
import pandas as pd
import sys
from nytimes_scraper.nyt_api import NytApi
from nytimes_scraper.articles import fetch_articles_by_month, articles_to_df
from nytimes_scraper.comments import fetch_comments, fetch_comments_by_article, comments_to_df

api = NytApi('lz3ClJ5oTuckGZAhiew2IAJ6t1qTvgS3')

year = sys.argv[1]
month = sys.argv[2]

name = 'article_' + year + '_' + month + '.csv'

print(name)

# Fetch articles of a specific month
articles = fetch_articles_by_month(api, dt.date(int(year), int(month), 1))
article_df = articles_to_df(articles)

article_df.drop(['snippet', 'print_section', 'print_page', 'source', 'multimedia',
       'uri', 'html', 'text', 'headline.kicker', 'headline.content_kicker',
       'headline.print_headline', 'headline.name', 'headline.seo',
       'headline.sub', 'byline.original', 'byline.person',
       'byline.organization', 'subsection_name'], axis=1, inplace=True)

article_df.to_csv(name)

import newspaper
import feedparser
import csv

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

def scrape_news_from_feed(feed_url):
    articles = []
    feed = feedparser.parse(feed_url, agent=USER_AGENT)

    config = newspaper.Config()
    config.browser_user_agent = USER_AGENT

    for entry in feed.entries:
        article = newspaper.Article(entry.link, config=config)
        article.download()
        article.parse()

        articles.append({
            "title": article.title,
            "author": article.authors,
            "publish_date": article.publish_date,
            "content": article.text
        })

    with open("news.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["title", "author", "publish_date", "content"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)
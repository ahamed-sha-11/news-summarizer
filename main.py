from news import scrape_news_from_feed
from summarizer import summarize_latest_news

FEEDS = {
    "1": ("Top Stories", "https://feeds.bbci.co.uk/news/rss.xml"),
    "2": ("Tamil Nadu", "https://www.thehindu.com/news/national/tamil-nadu/feeder/default.rss"),
    "3": ("International", "https://www.thehindu.com/news/international/feeder/default.rss"),
    "4": ("Sport", "https://www.thehindu.com/sport/feeder/default.rss"),
}

if __name__ == "__main__":
    while True:
        print()
        print("What news do you want today?")
        for key, (name, _) in FEEDS.items():
            print(f"  {key} : {name}")
        print("  5 : Exit")
        option = input("Choose: ")

        if option == "5":
            break

        if option not in FEEDS:
            print("Invalid option")
            continue

        name, url = FEEDS[option]
        print(f"\nGetting your {name} news ...\n")

        scrape_news_from_feed(url)
        summarize_latest_news()
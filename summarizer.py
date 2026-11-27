import os
import textwrap

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

class News(BaseModel):
    title: str = Field(description="One line summary of the news")
    authors: list[str] = Field(description="The authors of the news")
    publish_date: datetime | None = Field(default=None, description="Date and time the news was published")
    content: str = Field(description="The content of the news")

class NewsResponse(BaseModel):
    news: list[News] = Field(description="all news summarized")


def summarize_latest_news():
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=gemini_api_key)

        df = pd.read_csv("news.csv")
        news = df.to_string(index=False)

        prompt = (
            "You are a professional news summarizer. "
            "Summarize the following news in bullet points:\n\n"
            + news
        )

        interaction = client.interactions.create(
            model="gemini-3.1-flash-lite",
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": NewsResponse.model_json_schema()
            },
        )



        news_summary = NewsResponse.model_validate_json(interaction.output_text)
        # print(news_summary)
        print_newspaper(news_summary)

    except FileNotFoundError:
        print("news.csv was not found. Scrape the news first.")
    except Exception as e:
        print(f"Failed to summarize news: {e}")

BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
RESET = "\033[0m"


def _story_lines(item, width):
    """Turn one story into a list of (text, style) lines that fit the width."""
    lines = []
    for line in textwrap.wrap(item.title.upper(), width):
        lines.append((line, BOLD))

    authors = ", ".join(item.authors) if item.authors else "Staff Reporter"
    byline = f"By {authors}"
    if item.publish_date:
        byline += f" | {item.publish_date.strftime('%d %b %Y')}"
    for line in textwrap.wrap(byline, width):
        lines.append((line, DIM + ITALIC))

    lines.append(("─" * width, DIM))
    for line in textwrap.wrap(item.content, width):
        lines.append((line, ""))
    return lines


def _styled(text, style, width):
    # Pad first, then add color, so the color codes don't break alignment
    padded = text.ljust(width)
    return f"{style}{padded}{RESET}" if style else padded


def print_newspaper(news_summary: NewsResponse, width: int = 80):
    today = datetime.now().strftime("%A, %d %B %Y")

    # Masthead
    print("=" * width)
    print(BOLD + "THE DAILY SUMMARY".center(width) + RESET)
    print(DIM + today.center(width) + RESET)
    print("=" * width)

    stories = news_summary.news
    if not stories:
        print("No news today.".center(width))
        return

    # Top story, full width
    print()
    print(BOLD + "✦  TOP STORY  ✦".center(width) + RESET)
    print()
    for text, style in _story_lines(stories[0], width):
        print(_styled(text, style, width))
    print()
    print("═" * width)

    # Remaining stories, two columns side by side
    col = (width - 3) // 2
    rest = stories[1:]
    for i in range(0, len(rest), 2):
        left = _story_lines(rest[i], col)
        right = _story_lines(rest[i + 1], col) if i + 1 < len(rest) else []

        print()
        for j in range(max(len(left), len(right))):
            l_text, l_style = left[j] if j < len(left) else ("", "")
            r_text, r_style = right[j] if j < len(right) else ("", "")
            print(_styled(l_text, l_style, col) + DIM + " │ " + RESET + _styled(r_text, r_style, col))
        print()
        print(DIM + "─" * width + RESET)

    # Footer
    print(DIM + "— END OF EDITION —".center(width) + RESET)
    print("=" * width)

if __name__ == "__main__":
    summarize_latest_news()
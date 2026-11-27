import json
import os
import textwrap

from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "gemini-3.1-flash-lite"
MAX_TOOL_ROUNDS = 10  # stop if the model keeps calling tools forever

# System prompt: who the model is and the rules it must follow.
SYSTEM_INSTRUCTION = """You are a professional news editor writing a daily newspaper digest.
Use the tools to read the news: call list_articles first, then get_article for each story you include.
Rules:
- Use only facts from the articles. Never invent authors, dates or details.
- title: a one-line headline. content: a 2-3 sentence summary.
- Put the most important story first.
- Skip duplicate stories and articles with empty content."""

# User prompt: the actual request for this run.
USER_PROMPT = "Summarize today's news."

# Sampling settings. Gemini 3 models are tuned for temperature 1.0, and Google
# advises against lowering it (it can cause looping or worse answers). On older
# models, a low temperature such as 0.2 suits factual summaries.
GENERATION_CONFIG = {
    "temperature": 0.4,
    "top_p": 0.95,           # sample only from the top 95% of probability mass
    "max_output_tokens": 8192,
    "seed": 42,              # same input + seed -> more repeatable output
}

# Tools the model may decide to call. We run them and send back the results.
TOOLS = [
    {
        "type": "function",
        "name": "list_articles",
        "description": "List the scraped articles with their index, title and publish date.",
    },
    {
        "type": "function",
        "name": "get_article",
        "description": "Get the full title, authors, publish date and text of one article.",
        "parameters": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Article index from list_articles"},
            },
            "required": ["index"],
        },
    },
]

class News(BaseModel):
    title: str = Field(description="One line summary of the news")
    authors: list[str] = Field(description="The authors of the news")
    publish_date: datetime | None = Field(default=None, description="Date and time the news was published")
    content: str = Field(description="The content of the news")

class NewsResponse(BaseModel):
    news: list[News] = Field(description="all news summarized")


def list_articles(df: pd.DataFrame) -> list[dict]:
    articles = []
    for index, row in df.iterrows():
        article = {
            "index": index,
            "title": row["title"],
            "publish_date": row["publish_date"],
        }
        articles.append(article)
    return articles


def get_article(df: pd.DataFrame, index: int) -> dict:
    if index not in df.index:
        raise ValueError(f"No article with index {index}")
    row = df.loc[index]
    return {
        "title": row["title"],
        "authors": row["author"],
        "publish_date": row["publish_date"],
        "content": row["content"],
    }


def run_tool(df: pd.DataFrame, call) -> dict:
    """Run one function call from the model and build the function_result step."""
    output = ""
    is_error = False

    try:
        if call.name == "list_articles":
            result = list_articles(df)
        elif call.name == "get_article":
            article_index = int(call.arguments["index"])
            result = get_article(df, article_index)
        else:
            raise ValueError(f"Unknown tool: {call.name}")

        output = json.dumps(result, default=str)
    except Exception as e:
        # Tell the model what went wrong so it can recover, instead of crashing
        output = f"Error: {e}"
        is_error = True

    return {
        "type": "function_result",
        "call_id": call.id,
        "name": call.name,
        "result": output,
        "is_error": is_error,
    }


def start_conversation(client: genai.Client, user_prompt: str):
    """Send the first message to the model."""
    return client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=GENERATION_CONFIG,
        tools=TOOLS,
        response_format=get_response_format(),
        input=user_prompt,
    )


def send_tool_results(client: genai.Client, previous_interaction_id: str, tool_results: list[dict]):
    """Send our tool results back, continuing the same conversation."""
    return client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=GENERATION_CONFIG,
        tools=TOOLS,
        response_format=get_response_format(),
        input=tool_results,
        previous_interaction_id=previous_interaction_id,
    )


def get_response_format() -> dict:
    """Ask the model to reply with JSON that matches the NewsResponse class."""
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": NewsResponse.model_json_schema(),
    }


def find_new_function_calls(interaction, answered_call_ids: set) -> list:
    """Return the function calls in the model's reply that we have not answered yet."""
    new_calls = []

    if interaction.steps is None:
        return new_calls

    for step in interaction.steps:
        if step.type != "function_call":
            continue
        if step.id in answered_call_ids:
            continue
        new_calls.append(step)

    return new_calls


def summarize_latest_news():
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=gemini_api_key)

        # Read everything as text; empty cells become "" instead of NaN
        df = pd.read_csv("news.csv", dtype=str, keep_default_na=False)

        interaction = start_conversation(client, USER_PROMPT)

        # Tool loop: while the model asks for function calls, run them and send the results back
        answered_call_ids = set()
        round_number = 0

        while True:
            function_calls = find_new_function_calls(interaction, answered_call_ids)

            # No tool requests left, so this reply is the model's final answer
            if len(function_calls) == 0:
                break

            round_number = round_number + 1
            if round_number > MAX_TOOL_ROUNDS:
                raise RuntimeError(f"Model was still calling tools after {MAX_TOOL_ROUNDS} rounds")

            tool_results = []
            for call in function_calls:
                print(f"{DIM}  [tool] {call.name}({call.arguments}){RESET}")
                result = run_tool(df, call)
                tool_results.append(result)
                answered_call_ids.add(call.id)

            interaction = send_tool_results(client, interaction.id, tool_results)

        news_summary = NewsResponse.model_validate_json(interaction.output_text)
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
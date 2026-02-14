#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, List, Optional, Set

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from dotenv import load_dotenv

from summarizer import BaseSummarizer, get_summarizer, truncate_by_sentences

BASE_URL = "https://tech.onliner.by/"
ARTICLE_URL_RE = re.compile(r"https://tech\.onliner\.by/\d{4}/\d{2}/\d{2}/[\w\-]+")
ARTICLE_DATE_RE = re.compile(r"https://tech\.onliner\.by/(\d{4})/(\d{2})/(\d{2})/[\w\-]+")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
TELEGRAM_CAPTION_LIMIT = 1024
LARGE_ARTICLE_THRESHOLD = 1500
STATE_MAX_AGE_DAYS = 7
UNWANTED_BODY_PATTERNS = [
    re.compile(r"есть\s+о\s+чем\s+рассказать\?", re.IGNORECASE),
    re.compile(r"пишите\s+в\s+наш\s+телеграм-бот", re.IGNORECASE),
    re.compile(r"это\s+анонимно\s+и\s+быстро", re.IGNORECASE),
]


@dataclass
class Article:
    url: str
    title: str
    published_at: Optional[datetime]
    author: Optional[str]
    views: Optional[str]
    header_image: Optional[str]
    body_paragraphs: List[str]


class OnlinerClient:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def fetch_listing_urls(self) -> List[str]:
        logging.info("Fetching listing page: %s", BASE_URL)
        response = self.session.get(BASE_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        urls = set()
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/"):
                href = f"https://tech.onliner.by{href}"
            if ARTICLE_URL_RE.match(href):
                urls.add(href)
        logging.info("Found %d article links", len(urls))
        return sorted(urls)

    def parse_article(self, url: str) -> Article:
        logging.info("Fetching article: %s", url)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        return Article(
            url=url,
            title=self._extract_title(soup),
            published_at=self._extract_published_at(soup),
            author=self._extract_author(soup),
            views=self._extract_views(soup),
            header_image=self._extract_header_image(soup),
            body_paragraphs=self._extract_body_paragraphs(soup),
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        meta = soup.find("meta", property="og:title")
        if meta and meta.get("content"):
            return meta["content"].strip()
        return "Без заголовка"

    @staticmethod
    def _extract_header_image(soup: BeautifulSoup) -> Optional[str]:
        meta = soup.find("meta", property="og:image")
        if meta and meta.get("content"):
            return meta["content"].strip()
        image = soup.find("img")
        if image and image.get("src"):
            return image["src"].strip()
        return None

    @staticmethod
    def _extract_published_at(soup: BeautifulSoup) -> Optional[datetime]:
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            return dateparser.parse(time_tag["datetime"])
        meta = soup.find("meta", property="article:published_time")
        if meta and meta.get("content"):
            return dateparser.parse(meta["content"])
        return None

    @staticmethod
    def _extract_author(soup: BeautifulSoup) -> Optional[str]:
        author = soup.find("a", attrs={"rel": "author"})
        if author and author.get_text(strip=True):
            return author.get_text(strip=True)
        meta = soup.find("meta", attrs={"name": "author"})
        if meta and meta.get("content"):
            return meta["content"].strip()
        return None

    @staticmethod
    def _extract_views(soup: BeautifulSoup) -> Optional[str]:
        view_tag = soup.find(class_=re.compile(r"views", re.IGNORECASE))
        if view_tag and view_tag.get_text(strip=True):
            return view_tag.get_text(strip=True)
        return None

    @staticmethod
    def _extract_body_paragraphs(soup: BeautifulSoup) -> List[str]:
        body = soup.find(class_=re.compile(r"news-text|news-body|article-body"))
        if not body:
            body = soup.find("article")

        paragraphs: List[str] = []
        if body:
            for paragraph in body.find_all("p"):
                text = paragraph.get_text(" ", strip=True)
                if text and not contains_unwanted_body_text(text):
                    paragraphs.append(text)

        if not paragraphs:
            description = soup.find("meta", attrs={"name": "description"})
            if description and description.get("content"):
                fallback_text = description["content"].strip()
                if not contains_unwanted_body_text(fallback_text):
                    paragraphs.append(fallback_text)
        return paragraphs


class TelegramPublisher:
    def __init__(self, token: str, channel: str, summarizer: BaseSummarizer) -> None:
        self.channel = channel
        self.summarizer = summarizer
        self.base_url = f"https://api.telegram.org/bot{token}"

    def post_article(self, article: Article) -> None:
        post_text = build_post(article, self.summarizer)
        payload = {
            "chat_id": self.channel,
            "disable_web_page_preview": True,
            "parse_mode": "HTML",
        }

        if article.header_image:
            endpoint = f"{self.base_url}/sendPhoto"
            payload["photo"] = article.header_image
            payload["caption"] = post_text
        else:
            endpoint = f"{self.base_url}/sendMessage"
            payload["text"] = post_text

        response = requests.post(endpoint, data=payload, timeout=30)
        response.raise_for_status()


def build_post(article: Article, summarizer: BaseSummarizer) -> str:
    title = article.title.strip() or "Без заголовка"
    original_body = "\n\n".join(article.body_paragraphs).strip()
    if not original_body:
        original_body = "Текст статьи отсутствует."

    source_label = "Подробнее..." if len(original_body) > LARGE_ARTICLE_THRESHOLD else "Источник"
    source_html = build_source_link(article.url, source_label)

    if post_visible_len(title, original_body, source_label) <= TELEGRAM_CAPTION_LIMIT:
        return compose_post(escape_html(title, keep_quotes=False), escape_html(original_body), source_html)

    try:
        summary = summarizer.summarize(original_body, 950)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Initial summarization failed: %s", exc)
        summary = truncate_by_sentences(original_body, 950)

    iteration = 0
    while post_visible_len(title, summary, source_label) > TELEGRAM_CAPTION_LIMIT and iteration < 3:
        new_limit = 900 - iteration * 50
        try:
            summary = summarizer.summarize(summary, new_limit)
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Refine summarization failed: %s", exc)
            summary = truncate_by_sentences(summary, new_limit)
        iteration += 1

    if post_visible_len(title, summary, source_label) > TELEGRAM_CAPTION_LIMIT:
        max_body_len = max(100, TELEGRAM_CAPTION_LIMIT - len(title) - len(source_label) - 4)
        summary = truncate_by_sentences(summary, max_body_len)

    if post_visible_len(title, summary, source_label) > TELEGRAM_CAPTION_LIMIT:
        body_limit = max(20, TELEGRAM_CAPTION_LIMIT - len(title) - len(source_label) - 4)
        summary = summary[:body_limit].rstrip()

    return compose_post(
        escape_html(title, keep_quotes=False),
        escape_html(summary),
        source_html,
    )


def compose_post(title: str, body: str, source: str) -> str:
    return f"<b>{title}</b>\n\n{body}\n\n{source}"


def build_source_link(url: str, label: str) -> str:
    safe_url = escape_html(url)
    safe_label = escape_html(label, keep_quotes=False)
    return f'<a href="{safe_url}">{safe_label}</a>'


def post_visible_len(title: str, body: str, source_label: str) -> int:
    return len(f"{title}\n\n{body}\n\n{source_label}")


def contains_unwanted_body_text(text: str) -> bool:
    normalized = " ".join(text.split())
    return any(pattern.search(normalized) for pattern in UNWANTED_BODY_PATTERNS)


def escape_html(text: str, keep_quotes: bool = True) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if not keep_quotes:
        return escaped
    return escaped.replace('"', "&quot;")


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"seen": []}
    with open(path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or "seen" not in state:
        return {"seen": []}
    return state


def save_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def load_seen_urls(path: str) -> Set[str]:
    state = load_state(path)
    seen = state.get("seen", [])
    if not isinstance(seen, list):
        return set()
    valid_urls = {url for url in seen if isinstance(url, str) and ARTICLE_URL_RE.match(url)}
    return prune_old_seen_urls(valid_urls)


def persist_seen_urls(path: str, seen: Set[str]) -> None:
    cleaned = prune_old_seen_urls(seen)
    save_state(path, {"seen": sorted(cleaned)})


def prune_old_seen_urls(seen_urls: Set[str]) -> Set[str]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=STATE_MAX_AGE_DAYS)
    cleaned: Set[str] = set()
    for url in seen_urls:
        article_date = extract_article_date(url)
        if article_date is None or article_date >= cutoff:
            cleaned.add(url)
    return cleaned


def extract_article_date(url: str) -> Optional[date]:
    match = ARTICLE_DATE_RE.match(url)
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def filter_recent_articles(articles: Iterable[Article], hours: int = 24) -> List[Article]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent = []
    for article in articles:
        if article.published_at is None:
            continue
        published = article.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published >= cutoff:
            recent.append(article)
    return recent


def run_last24h(
    client: OnlinerClient,
    publisher: TelegramPublisher,
    state_file: str,
) -> None:
    seen = load_seen_urls(state_file)
    persist_seen_urls(state_file, seen)

    urls = client.fetch_listing_urls()
    articles = [client.parse_article(url) for url in urls if url not in seen]
    recent_articles = filter_recent_articles(articles)
    for article in sorted(
        recent_articles, key=lambda item: item.published_at or datetime.now(timezone.utc)
    ):
        publisher.post_article(article)
        seen.add(article.url)
        persist_seen_urls(state_file, seen)
        time.sleep(1)


def run_watch(
    client: OnlinerClient,
    publisher: TelegramPublisher,
    interval: int,
    state_file: str,
) -> None:
    seen = load_seen_urls(state_file)
    persist_seen_urls(state_file, seen)

    while True:
        try:
            urls = client.fetch_listing_urls()
            new_urls = [url for url in urls if url not in seen]
            if new_urls:
                logging.info("Found %d new articles", len(new_urls))
            for url in new_urls:
                article = client.parse_article(url)
                publisher.post_article(article)
                seen.add(url)
                persist_seen_urls(state_file, seen)
                time.sleep(1)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Error while fetching updates: %s", exc)
        logging.info("Sleeping for %d seconds", interval)
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Парсер раздела технологий onliner.by с публикацией в Telegram."
    )
    parser.add_argument("mode", choices=["last24h", "watch"], help="Режим работы")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Интервал проверки новых новостей (секунды)",
    )
    parser.add_argument(
        "--state-file",
        default="state.json",
        help="Файл состояния для хранения уже опубликованных ссылок",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Уровень логирования",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    channel = os.getenv("TELEGRAM_CHANNEL")
    if not token or not channel:
        raise SystemExit("Нужны TELEGRAM_BOT_TOKEN и TELEGRAM_CHANNEL в окружении")

    client = OnlinerClient()
    summarizer = get_summarizer()
    publisher = TelegramPublisher(token=token, channel=channel, summarizer=summarizer)

    if args.mode == "last24h":
        run_last24h(client, publisher, args.state_file)
    else:
        run_watch(client, publisher, args.interval, args.state_file)


if __name__ == "__main__":
    main()

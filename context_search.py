# Retrieves and ranks relevant news articles to provide real-world context for unusual patterns

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen

import pandas as pd


@dataclass
class NewsContext:
    should_search: bool
    trigger_reasons: list[str]
    articles: list[dict[str, Any]]
    summary_points: list[str]


def _strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def should_search_context(
    recent_df: pd.DataFrame,
    diagnostics: dict[str, Any],
    metrics: dict[str, Any],
    trend_trigger_pct: float = 3.0,
    one_day_trigger_pct: float = 2.5,
    mape_trigger_pct: float = 5.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    anomaly_count = int(diagnostics.get("anomaly_count", 0) or 0)
    recent_trend = _safe_float(diagnostics.get("recent_trend_5d_pct", 0.0))
    mape = _safe_float(metrics.get("mape", 999.0))

    if anomaly_count > 0:
        reasons.append(f"anomaly_count={anomaly_count}")

    if abs(recent_trend) >= trend_trigger_pct:
        reasons.append(f"recent_trend_5d_pct={recent_trend:.2f}")

    if len(recent_df) >= 2:
        last = float(recent_df["value"].iloc[-1])
        prev = float(recent_df["value"].iloc[-2])
        if prev != 0:
            one_day_move = (last / prev - 1.0) * 100.0
            if abs(one_day_move) >= one_day_trigger_pct:
                reasons.append(f"one_day_move_pct={one_day_move:.2f}")

    if mape >= mape_trigger_pct:
        reasons.append(f"mape={mape:.2f}")

    return (len(reasons) > 0), reasons


def _google_news_rss(query: str, when_days: int = 21, limit: int = 10) -> list[dict[str, Any]]:
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}+when:{when_days}d&hl=en-US&gl=US&ceid=US:en"
    )

    with urlopen(url, timeout=20) as resp:
        xml_bytes = resp.read()

    root = ET.fromstring(xml_bytes)
    items = root.findall(".//item")

    results: list[dict[str, Any]] = []
    for item in items[:limit]:
        title = _strip_html(item.findtext("title", default=""))
        link = item.findtext("link", default="")
        pub_date = item.findtext("pubDate", default="")
        description = _strip_html(item.findtext("description", default=""))

        published_at = None
        if pub_date:
            try:
                published_at = parsedate_to_datetime(pub_date).isoformat()
            except Exception:
                published_at = None

        results.append(
            {
                "title": title,
                "link": link,
                "published_at": published_at,
                "snippet": description[:300],
            }
        )

    return results


def _default_company_name(symbol: str) -> str | None:
    symbol = (symbol or "").upper().strip()
    common = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet",
        "GOOG": "Alphabet",
        "AMZN": "Amazon",
        "META": "Meta",
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        "JPM": "JPMorgan",
        "XOM": "Exxon Mobil",
    }
    return common.get(symbol)


def _default_aliases(symbol: str, company_name: str | None) -> list[str]:
    symbol = (symbol or "").upper().strip()
    aliases: list[str] = []

    if company_name:
        aliases.append(company_name)

    special = {
        "AAPL": ["Apple", "iPhone maker"],
        "MSFT": ["Microsoft"],
        "GOOGL": ["Alphabet", "Google"],
        "GOOG": ["Alphabet", "Google"],
        "AMZN": ["Amazon"],
        "META": ["Meta", "Facebook"],
        "NVDA": ["NVIDIA"],
        "TSLA": ["Tesla"],
        "JPM": ["JPMorgan", "JP Morgan"],
        "XOM": ["Exxon Mobil", "Exxon"],
    }
    aliases.extend(special.get(symbol, []))

    # dedupe while preserving order
    seen = set()
    out = []
    for x in aliases:
        x2 = x.strip()
        if x2 and x2.lower() not in seen:
            seen.add(x2.lower())
            out.append(x2)
    return out


def _default_sector_keywords(symbol: str) -> list[str]:
    symbol = (symbol or "").upper().strip()

    tech = ["software", "cloud", "chip", "device", "ai", "hardware", "platform"]
    bank = ["deposit", "loan", "credit", "fed", "margin", "banking"]
    energy = ["oil", "gas", "production", "refining", "opec", "energy"]
    consumer = ["retail", "consumer", "product", "store", "demand"]
    industrial = ["factory", "manufacturing", "supply chain", "orders", "industrial"]

    if symbol in {"AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META"}:
        return tech
    if symbol in {"JPM", "BAC", "WFC", "C", "GS", "MS"}:
        return bank
    if symbol in {"XOM", "CVX", "COP", "SLB"}:
        return energy
    if symbol in {"AMZN", "WMT", "TGT", "COST", "NKE"}:
        return consumer
    return industrial


def get_symbol_context(
    symbol: str,
    company_name: str | None = None,
    sector_keywords: list[str] | None = None,
) -> dict[str, Any]:
    resolved_company = company_name or _default_company_name(symbol)
    resolved_sector_keywords = sector_keywords or _default_sector_keywords(symbol)
    resolved_aliases = _default_aliases(symbol, resolved_company)

    return {
        "symbol": symbol.upper().strip(),
        "company_name": resolved_company,
        "aliases": resolved_aliases,
        "sector_keywords": resolved_sector_keywords,
    }


def _build_query(symbol: str, company_name: str | None, aliases: list[str], sector_keywords: list[str]) -> str:
    # Make entity terms mandatory in spirit by centering the query on them.
    entity_terms = [symbol]
    if company_name:
        entity_terms.append(company_name)
    entity_terms.extend(aliases[:2])

    generic_event_terms = [
        "earnings",
        "guidance",
        "outlook",
        "price target",
        "upgrade",
        "downgrade",
        "product",
        "launch",
    ]

    pieces = entity_terms + generic_event_terms[:4] + sector_keywords[:1]
    pieces = [x for x in pieces if x]

    # dedupe while preserving order
    deduped = list(dict.fromkeys(pieces))
    return " OR ".join(deduped)


def _score_articles(
    articles: list[dict[str, Any]],
    symbol: str,
    company_name: str | None = None,
    aliases: list[str] | None = None,
    sector_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []

    entity_terms = [symbol.lower()]
    if company_name:
        entity_terms.append(company_name.lower())
    entity_terms.extend([x.lower() for x in (aliases or []) if x])

    generic_event_terms = [
        "earnings",
        "guidance",
        "outlook",
        "price target",
        "target",
        "upgrade",
        "downgrade",
        "product",
        "launch",
        "acquisition",
        "merger",
        "lawsuit",
        "regulation",
        "tariff",
        "demand",
    ]

    sector_terms = [x.lower() for x in (sector_keywords or [])]

    trusted_sources = [
        "reuters",
        "bloomberg",
        "wall street journal",
        "wsj",
        "financial times",
        "yahoo finance",
        "marketwatch",
        "investor relations",
    ]

    for article in articles:
        title = (article.get("title") or "").lower()
        snippet = (article.get("snippet") or "").lower()
        blob = f"{title} {snippet}"

        # Strong company/ticker relevance.
        entity_matches = sum(1 for term in entity_terms if term and term in blob)
        entity_score = 4 * entity_matches

        # Slight extra bonus if entity is in title specifically.
        title_entity_bonus = 3 if any(term and term in title for term in entity_terms) else 0

        # HARD FILTER: skip articles with no entity match at all.
        if entity_score == 0 and title_entity_bonus == 0:
            continue

        event_score = 1 * sum(1 for term in generic_event_terms if term in blob)
        sector_score = 1 * sum(1 for term in sector_terms if term in blob)
        source_score = 2 * sum(1 for src in trusted_sources if src in blob)

        total_score = entity_score + title_entity_bonus + event_score + sector_score + source_score

        enriched = dict(article)
        enriched["score"] = total_score
        enriched["entity_score"] = entity_score
        enriched["title_entity_bonus"] = title_entity_bonus
        enriched["event_score"] = event_score
        enriched["sector_score"] = sector_score
        enriched["source_score"] = source_score
        scored.append(enriched)

    return sorted(
        scored,
        key=lambda x: (x["score"], x.get("published_at") or ""),
        reverse=True,
    )[:5]


def _summarize_articles(symbol: str, articles: list[dict[str, Any]]) -> list[str]:
    if not articles:
        return [f"No strong recent news context found for {symbol}."]

    points: list[str] = []
    for article in articles[:3]:
        date_part = ""
        if article.get("published_at"):
            date_part = f"{str(article['published_at'])[:10]}: "
        points.append(f"{date_part}{article.get('title', '')}")
    return points


def find_news_context(
    symbol: str,
    recent_df: pd.DataFrame,
    diagnostics: dict[str, Any],
    metrics: dict[str, Any],
    enabled: bool = True,
    when_days: int = 21,
    company_name: str | None = None,
    sector_keywords: list[str] | None = None,
) -> NewsContext:
    if not enabled:
        return NewsContext(False, [], [], ["Context search disabled."])

    should_search, trigger_reasons = should_search_context(recent_df, diagnostics, metrics)
    if not should_search:
        return NewsContext(False, [], [], ["No news search triggered."])

    symbol_ctx = get_symbol_context(
        symbol=symbol,
        company_name=company_name,
        sector_keywords=sector_keywords,
    )

    query = _build_query(
        symbol=symbol_ctx["symbol"],
        company_name=symbol_ctx["company_name"],
        aliases=symbol_ctx["aliases"],
        sector_keywords=symbol_ctx["sector_keywords"],
    )

    try:
        raw_articles = _google_news_rss(query=query, when_days=when_days, limit=10)
        articles = _score_articles(
            raw_articles,
            symbol=symbol_ctx["symbol"],
            company_name=symbol_ctx["company_name"],
            aliases=symbol_ctx["aliases"],
            sector_keywords=symbol_ctx["sector_keywords"],
        )
        summary_points = _summarize_articles(symbol_ctx["symbol"], articles)
        return NewsContext(True, trigger_reasons, articles, summary_points)
    except Exception as exc:
        return NewsContext(
            True,
            trigger_reasons,
            [],
            [f"News search attempted but failed: {exc}"],
        )

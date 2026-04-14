#!/usr/bin/env python3
"""Update README WakaTime total time by summing version badge durations.

This script parses WakaTime badge SVGs referenced by v1-vN rows in README,
converts each duration to minutes, computes the total, and rewrites the
`| **Total** | ... |` table row.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


VERSION_ROW_RE = re.compile(r"^\|\s*(v\d+)\s*\|", re.IGNORECASE)
BADGE_URL_RE = re.compile(r"https://wakatime\.com/badge/[^\s)]+\.svg", re.IGNORECASE)
TOTAL_S_ROW_RE = re.compile(r"^\|\s*\*\*Total Sums\*\*\s*\|.*\|$", re.IGNORECASE | re.MULTILINE)
TOTAL_ROW_RE = re.compile(r"^\|\s*\*\*Total\*\*\s*\|.*\|$", re.IGNORECASE | re.MULTILINE)
SVG_TEXT_RE = re.compile(r"<text[^>]*>([^<]+)</text>", re.IGNORECASE)
TIME_TOKEN_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>days?|day|d|hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)\b",
    re.IGNORECASE,
)


def fetch_text(url: str, timeout_seconds: float) -> str:
    request = Request(url, headers={"User-Agent": "prert-wakatime-updater/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - fixed trusted host pattern
        return response.read().decode("utf-8", errors="replace")


def extract_version_badges(readme_text: str) -> List[Tuple[str, str]]:
    badges: List[Tuple[str, str]] = []
    for raw_line in readme_text.splitlines():
        line = raw_line.rstrip()
        row_match = VERSION_ROW_RE.match(line)
        if not row_match:
            continue
        version = row_match.group(1).lower()
        url_match = BADGE_URL_RE.search(line)
        if not url_match:
            continue
        badges.append((version, url_match.group(0)))

    if not badges:
        raise RuntimeError("No WakaTime version badge URLs were found in README.")
    return badges


def extract_time_text_from_svg(svg_text: str) -> str:
    texts = [html.unescape(chunk).strip() for chunk in SVG_TEXT_RE.findall(svg_text)]
    candidates = [chunk for chunk in texts if chunk and chunk.lower() != "wakatime" and any(ch.isdigit() for ch in chunk)]
    if not candidates:
        raise RuntimeError("Could not parse time text from WakaTime badge SVG.")
    return candidates[-1]

def extract_research_minutes_by_version(readme_text: str) -> Dict[str, int]:
    results: Dict[str, int] = {}
    for raw_line in readme_text.splitlines():
        line = raw_line.rstrip()
        row_match = VERSION_ROW_RE.match(line)
        if not row_match:
            continue

        version = row_match.group(1).lower()
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 3:
            results[version] = 0
            continue

        research_text = cells[2]
        if not research_text:
            results[version] = 0
            continue

        results[version] = parse_duration_minutes(research_text)

    return results

def parse_duration_minutes(time_text: str) -> int:
    total_minutes = 0.0
    matched = False
    for match in TIME_TOKEN_RE.finditer(time_text):
        matched = True
        value = float(match.group("value"))
        unit = match.group("unit").lower()

        if unit in {"day", "days", "d"}:
            total_minutes += value * 24 * 60
        elif unit in {"hour", "hours", "hr", "hrs", "h"}:
            total_minutes += value * 60
        elif unit in {"minute", "minutes", "min", "mins", "m"}:
            total_minutes += value
        elif unit in {"second", "seconds", "sec", "secs", "s"}:
            total_minutes += value / 60.0

    if not matched:
        raise RuntimeError(f"Could not parse duration tokens from '{time_text}'.")
    return int(round(total_minutes))


def format_minutes(total_minutes: int) -> str:
    hours, minutes = divmod(total_minutes, 60)

    parts: List[str] = []
    if hours:
        parts.append(f"{hours} hr" + ("s" if hours != 1 else ""))
    if minutes or not parts:
        parts.append(f"{minutes} min" + ("s" if minutes != 1 else ""))
    return " ".join(parts)


def _replace_row(readme_text: str, pattern: re.Pattern[str], new_row: str, label: str) -> str:
    if not pattern.search(readme_text):
        raise RuntimeError(f"README does not contain a '{label}' row to update.")
    return pattern.sub(new_row, readme_text, count=1)


def update_total_rows(
    readme_text: str,
    total_coding_text: str,
    total_research_text: str,
    total_text: str,
) -> str:
    updated = readme_text
    updated = _replace_row(
        updated,
        TOTAL_S_ROW_RE,
        f"|  **Total Sums**  | **{total_coding_text}** | **{total_research_text}** |",
        "| **Total Sums** | ... |",
    )
    updated = _replace_row(
        updated,
        TOTAL_ROW_RE,
        f"|     **Total**      | **{total_text}** |                     |",
        "| **Total** | ... |",
    )
    return updated


def compute_badge_minutes(badges: Iterable[Tuple[str, str]], timeout_seconds: float) -> Dict[str, int]:
    results: Dict[str, int] = {}
    for version, url in badges:
        svg = fetch_text(url=url, timeout_seconds=timeout_seconds)
        time_text = extract_time_text_from_svg(svg)
        results[version] = parse_duration_minutes(time_text)
    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update README WakaTime total by summing version badge times.")
    parser.add_argument("--readme", type=Path, default=Path("README.md"), help="Path to README file.")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds for each badge fetch.")
    parser.add_argument("--write", action="store_true", help="Write updates to README in place.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readme_path = args.readme
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")

    original = readme_path.read_text(encoding="utf-8")
    badges = extract_version_badges(original)
    research_minutes_by_version = extract_research_minutes_by_version(original)
    print(f"Found {len(badges)} version badges and {len(research_minutes_by_version)} research-time rows")
    
    try:
        versions_minutes = compute_badge_minutes(badges=badges, timeout_seconds=float(args.timeout))
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to fetch or parse WakaTime badges: {exc}") from exc

    research_minutes = sum(research_minutes_by_version.get(version, 0) for version, _ in badges)
    value_minutes = sum(versions_minutes.values())
    total_minutes = value_minutes + research_minutes
    total_coding_text = format_minutes(value_minutes)
    total_research_text = format_minutes(research_minutes)
    total_text = format_minutes(total_minutes)
    updated = update_total_rows(
        original,
        total_coding_text=total_coding_text,
        total_research_text=total_research_text,
        total_text=total_text,
    )

    print("Parsed version durations:")
    for version in sorted(versions_minutes):
        research_for_version = research_minutes_by_version.get(version, 0)
        print(
            f"- {version}: coding={format_minutes(versions_minutes[version])} ({versions_minutes[version]} mins), "
            f"research={format_minutes(research_for_version)} ({research_for_version} mins)"
        )
    print(f"Research total: {format_minutes(research_minutes)} ({research_minutes} mins)")
    print(f"Computed total: {total_text} ({total_minutes} mins)")

    if args.write:
        if updated != original:
            readme_path.write_text(updated, encoding="utf-8")
            print(f"Updated {readme_path}")
        else:
            print("README total row already up to date.")
    else:
        print("Dry run complete. Re-run with --write to apply changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
DEFAULT_FORM = "10-K"
DEFAULT_OUTPUT_DIR = Path("data/documents/sec_filings")
DEFAULT_MANIFEST_PATH = Path("data/sec_filings_manifest.json")
DEFAULT_USER_AGENT = "rag-knowledge-qa-system/0.1 (educational project; contact@example.com)"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 0.25


@dataclass(frozen=True)
class Company:
    cik: int
    ticker: str
    title: str


@dataclass(frozen=True)
class Filing:
    ticker: str
    company_name: str
    cik: int
    form: str
    accession_number: str
    primary_document: str
    filing_date: str
    report_date: str
    source_url: str


@dataclass(frozen=True)
class DownloadedFiling:
    ticker: str
    company_name: str
    cik: int
    form: str
    accession_number: str
    primary_document: str
    filing_date: str
    report_date: str
    source_url: str
    local_path: str
    downloaded: bool


def fetch_json(url: str, user_agent: str) -> dict:
    payload = fetch_bytes(url, user_agent=user_agent)
    return json.loads(payload.decode("utf-8"))


def fetch_bytes(url: str, user_agent: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json,text/html,application/xhtml+xml,text/plain,*/*",
        },
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def parse_company_tickers(payload: dict) -> dict[str, Company]:
    companies: dict[str, Company] = {}
    for entry in payload.values():
        ticker = str(entry["ticker"]).upper()
        companies[ticker] = Company(cik=int(entry["cik_str"]), ticker=ticker, title=str(entry["title"]))
    return companies


def build_archive_url(cik: int, accession_number: str, primary_document: str) -> str:
    accession_path = accession_number.replace("-", "")
    return (
        f"{SEC_ARCHIVES_BASE_URL}/{int(cik)}/{accession_path}/"
        f"{quote(primary_document, safe='')}"
    )


def select_recent_filings(company: Company, submission: dict, form: str, limit: int) -> list[Filing]:
    recent = submission.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    selected: list[Filing] = []
    wanted_form = form.upper()
    for index, filing_form in enumerate(forms):
        if str(filing_form).upper() != wanted_form:
            continue

        accession_number = _get_at(accession_numbers, index)
        primary_document = _get_at(primary_documents, index)
        if not accession_number or not primary_document:
            continue

        selected.append(
            Filing(
                ticker=company.ticker,
                company_name=company.title,
                cik=company.cik,
                form=str(filing_form),
                accession_number=accession_number,
                primary_document=primary_document,
                filing_date=_get_at(filing_dates, index),
                report_date=_get_at(report_dates, index),
                source_url=build_archive_url(company.cik, accession_number, primary_document),
            )
        )
        if len(selected) >= limit:
            break

    return selected


def output_path_for_filing(filing: Filing, output_dir: Path) -> Path:
    extension = Path(filing.primary_document).suffix.lower() or ".html"
    safe_form = _safe_file_part(filing.form)
    safe_date = _safe_file_part(filing.filing_date or "unknown-date")
    safe_accession = _safe_file_part(filing.accession_number)
    filename = f"{filing.ticker}_{safe_form}_{safe_date}_{safe_accession}{extension}"
    return output_dir / filing.ticker / filename


def download_latest_filings(
    tickers: list[str],
    form: str,
    limit: int,
    output_dir: Path,
    manifest_path: Path | None,
    user_agent: str,
    force: bool,
    delay_seconds: float,
) -> list[DownloadedFiling]:
    company_tickers = parse_company_tickers(fetch_json(SEC_COMPANY_TICKERS_URL, user_agent=user_agent))
    downloaded: list[DownloadedFiling] = []

    for raw_ticker in tickers:
        ticker = raw_ticker.upper()
        company = company_tickers.get(ticker)
        if company is None:
            raise ValueError(f"Ticker not found in SEC company tickers: {raw_ticker}")

        submission_url = SEC_SUBMISSIONS_URL.format(cik=company.cik)
        submission = fetch_json(submission_url, user_agent=user_agent)
        filings = select_recent_filings(company=company, submission=submission, form=form, limit=limit)
        if not filings:
            raise ValueError(f"No {form} filings found for ticker: {ticker}")

        for filing in filings:
            local_path = output_path_for_filing(filing, output_dir=output_dir)
            was_downloaded = False
            if force or not local_path.exists():
                time.sleep(max(delay_seconds, 0))
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(fetch_bytes(filing.source_url, user_agent=user_agent))
                was_downloaded = True

            downloaded.append(
                DownloadedFiling(
                    ticker=filing.ticker,
                    company_name=filing.company_name,
                    cik=filing.cik,
                    form=filing.form,
                    accession_number=filing.accession_number,
                    primary_document=filing.primary_document,
                    filing_date=filing.filing_date,
                    report_date=filing.report_date,
                    source_url=filing.source_url,
                    local_path=str(local_path),
                    downloaded=was_downloaded,
                )
            )

        time.sleep(max(delay_seconds, 0))

    if manifest_path is not None:
        write_manifest(downloaded, manifest_path)

    return downloaded


def write_manifest(downloaded: list[DownloadedFiling], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(item) for item in downloaded], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_user_agent(cli_user_agent: str | None) -> str:
    return cli_user_agent or os.getenv("SEC_USER_AGENT") or DEFAULT_USER_AGENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public SEC company filings for the local RAG corpus.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help=f"Ticker symbols to download. Default: {' '.join(DEFAULT_TICKERS)}",
    )
    parser.add_argument("--form", default=DEFAULT_FORM, help=f"SEC form type to download. Default: {DEFAULT_FORM}")
    parser.add_argument("--limit", type=int, default=1, help="Number of filings per ticker. Default: 1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for downloaded filing documents. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"JSON manifest path. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument("--no-manifest", action="store_true", help="Do not write a JSON download manifest.")
    parser.add_argument("--user-agent", help="SEC User-Agent header. Defaults to SEC_USER_AGENT or a project UA.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing local filing files.")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help=f"Delay between SEC requests. Default: {REQUEST_DELAY_SECONDS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    try:
        results = download_latest_filings(
            tickers=args.tickers,
            form=args.form,
            limit=args.limit,
            output_dir=args.output_dir,
            manifest_path=None if args.no_manifest else args.manifest_path,
            user_agent=build_user_agent(args.user_agent),
            force=args.force,
            delay_seconds=args.delay_seconds,
        )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise SystemExit(f"Failed to download SEC filings: {exc}") from exc

    downloaded_count = sum(1 for item in results if item.downloaded)
    reused_count = len(results) - downloaded_count
    print(f"SEC filings ready: downloaded={downloaded_count}, reused={reused_count}, total={len(results)}")
    for item in results:
        action = "downloaded" if item.downloaded else "reused"
        print(f"- {action}: {item.ticker} {item.form} {item.filing_date} -> {item.local_path}")


def _get_at(values: list, index: int) -> str:
    if index >= len(values) or values[index] is None:
        return ""
    return str(values[index])


def _safe_file_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


if __name__ == "__main__":
    main()

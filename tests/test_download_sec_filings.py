from pathlib import Path

from scripts.download_sec_filings import (
    Company,
    build_archive_url,
    output_path_for_filing,
    parse_company_tickers,
    select_recent_filings,
)


def test_parse_company_tickers_indexes_by_uppercase_ticker():
    payload = {
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }

    companies = parse_company_tickers(payload)

    assert companies["AAPL"] == Company(cik=320193, ticker="AAPL", title="Apple Inc.")
    assert companies["MSFT"].title == "MICROSOFT CORP"


def test_select_recent_filings_returns_matching_forms_in_order():
    company = Company(cik=320193, ticker="AAPL", title="Apple Inc.")
    submission = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-K", "10-Q", "10-K"],
                "accessionNumber": [
                    "0000320193-24-000120",
                    "0000320193-24-000123",
                    "0000320193-24-000080",
                    "0000320193-23-000106",
                ],
                "primaryDocument": ["aapl-8k.htm", "aapl-20240928.htm", "aapl-10q.htm", "aapl-20230930.htm"],
                "filingDate": ["2024-10-15", "2024-11-01", "2024-08-02", "2023-11-03"],
                "reportDate": ["2024-10-15", "2024-09-28", "2024-06-29", "2023-09-30"],
            }
        }
    }

    filings = select_recent_filings(company=company, submission=submission, form="10-K", limit=1)

    assert len(filings) == 1
    assert filings[0].ticker == "AAPL"
    assert filings[0].form == "10-K"
    assert filings[0].filing_date == "2024-11-01"
    assert filings[0].source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000123/aapl-20240928.htm"
    )


def test_output_path_for_filing_uses_ticker_subdirectory():
    company = Company(cik=320193, ticker="AAPL", title="Apple Inc.")
    filing = select_recent_filings(
        company=company,
        submission={
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "accessionNumber": ["0000320193-24-000123"],
                    "primaryDocument": ["aapl-20240928.htm"],
                    "filingDate": ["2024-11-01"],
                    "reportDate": ["2024-09-28"],
                }
            }
        },
        form="10-K",
        limit=1,
    )[0]

    assert output_path_for_filing(filing, Path("data/documents/sec_filings")) == Path(
        "data/documents/sec_filings/AAPL/AAPL_10-K_2024-11-01_0000320193-24-000123.htm"
    )


def test_build_archive_url_quotes_primary_document_path():
    assert build_archive_url(1, "0000000001-24-000001", "form 10-k.htm") == (
        "https://www.sec.gov/Archives/edgar/data/1/000000000124000001/form%2010-k.htm"
    )

import argparse
import logging
import time

from app.core.config import get_settings
from app.ingestion.jobs import process_next_job


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run queued RAG ingestion jobs.")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds to sleep when no job is queued.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()

    while True:
        job = process_next_job(settings)
        if job is not None:
            logger.info("ingestion_job_processed id=%s status=%s version=%s", job.id, job.status, job.target_index_version)
        elif args.once:
            logger.info("ingestion_job_none_queued")

        if args.once:
            return
        if job is None:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()

import arxiv
from backend.config import settings
from backend.paths import project_root
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


def download_recent_ai_papers(max_papers: int = 50) -> None:
    """Fetches the latest cs.AI papers from arXiv (newest first, capped at max_papers)."""
    target_dir = project_root() / settings.RAW_PDF_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving PDFs to: %s", target_dir.resolve())
    logger.info("Searching arXiv for %s recent cs.AI papers...", max_papers)

    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    for result in client.results(search):
        safe_title = "".join(
            [c for c in result.title if c.isalnum() or c == " "]
        ).rstrip()
        filename = f"{safe_title}.pdf"
        filepath = target_dir / filename

        if not filepath.exists():
            logger.info("Downloading: %s", result.title)
            try:
                result.download_pdf(dirpath=str(target_dir), filename=filename)
            except Exception as e:
                logger.error("Failed to download %s: %s", result.title, e)
        else:
            logger.info("Already have: %s", result.title)

    logger.info("Data ingestion complete.")


if __name__ == "__main__":
    download_recent_ai_papers(50)

import os
import arxiv
from backend.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

def download_recent_ai_papers(max_papers=50):
    """ Automatically fetches the latest cs.AI papers from arXiv """
    
    # absolute path
    project_root = os.getcwd()  
    target_dir = os.path.abspath(os.path.join(project_root, settings.RAW_PDF_DIR))
    
    # To ensure the folder exists
    os.makedirs(target_dir, exist_ok=True)
    
    logger.info(f"Saving PDFs to absolute path: {target_dir}")
    logger.info(f"Searching arXiv for {max_papers} recent cs.AI papers...")
    
    # Creating the search query for AI papers sorted by newest
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate  # For Recent Papers !!
    )

    # Loop through the results and download them
    for result in client.results(search):
        # Create a safe filename
        safe_title = "".join([c for c in result.title if c.isalpha() or c.isdigit() or c==' ']).rstrip() # Sanitization of filename
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(target_dir, filename)
        
        # Only download if we don't already have it.. Idempotent Downloads :-
        if not os.path.exists(filepath):
            logger.info(f"Downloading: {result.title}")
            try:
                result.download_pdf(dirpath=target_dir, filename=filename)
            except Exception as e:
                logger.error(f"Failed to download {result.title}: {e}")
        else:
            logger.info(f"Already have: {result.title}")

    logger.info(" Data ingestion complete!")

if __name__ == "__main__":
    download_recent_ai_papers(50)
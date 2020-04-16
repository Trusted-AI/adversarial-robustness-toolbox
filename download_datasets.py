import logging

from art.config import ART_DATA_PATH
from art.utils import get_file, art_datasets_urls

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

for file_name, url in art_datasets_urls.items():
    logger.info("Downloading {0} dataset".format(file_name))
    path = get_file(file_name, path=ART_DATA_PATH, url=url)

logger.info("Finished downloading datasets")


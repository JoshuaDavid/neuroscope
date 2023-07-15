import dotenv
import os
from pathlib import Path

dotenv.load_dotenv()

WEBSITE_VERSION = 4

CACHE_DIR = Path.home() / ("cache")
REPO_ROOT = Path.home() / ("hf_repos/")
OLD_CHECKPOINT_DIR = Path.home() / ("solu_project/solu_checkpoints/")
CHECKPOINT_DIR = Path.home() / ("solu_project/saved_models/")

MAKE_META_FILES = bool(os.getenv('MAKE_META_FILES', True))
WEBSITE_DIR     = os.getenv('WEBSITE_DIR', '/workspace/neuroscope/v{WEBSITE_VERSION}')
IN_IPYTHON      = False

import dotenv

dotenv.load_dotenv()

WEBSITE_VERSION = 4
MAKE_META_FILES = bool(os.getenv('MAKE_META_FILES', True))
WEBSITE_DIR     = os.getenv('WEBSITE_DIR', '/workspace/neuroscope/v{WEBSITE_VERSION}')

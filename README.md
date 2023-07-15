# Neuroscope
Accompanying codebase for neuroscope.io, a website for displaying max activating dataset examples for language model neurons

## Installation

1. Clone the repo
```sh
git clone https://github.com/neelnanda-io/Neuroscope.git
```
2. Make and use a virtual env (optional, but useful if you're going to be using this on a computer where other Python projects might be running with incompatible dependencies)
```sh
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```sh
pip install .
```
4. Override any configuration you want by creating a `.env` file (optional, you can see what config exists in `neuroscope/config.py`)

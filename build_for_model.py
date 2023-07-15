import os
import sys
import argparse

from neuroscope.config import (
    WEBSITE_DIR,
)

from neuroscope.make_neuroscope_index_pages import (
    gen_main_index_page,
    make_page_file,
)

parser = argparse.ArgumentParser(description="""
Create a folder of html pages, in the following structure

- index.html (lists all models)
- {model_name}/
    - index.html (lists all layers of that model)
    - {layer_number}/
        - {neuron_number}.html (shows the 20 max activations of that neuron across the dataset)
""")

parser.add_argument(
    '-m', '--model', type=str,
    help="Which model to make neuroscope pages for. See --list-models for a list of supported models."
)

args = parser.parse_args()

if args.model:
    model_name = args.model
    print(f"Building for {model_name}")
    main_index_html = gen_main_index_page([model_name])
    main_index_path = os.path.join(WEBSITE_DIR, "index.html")
    make_page_file(main_index_path, main_index_html)
    print(f"Wrote {main_index_path}")

import os
import sys
import argparse
from neuroscope.templates import (REDIRECT_TO_INDEX)
from neuroscope.config import (WEBSITE_DIR)

from neuroscope.make_neuroscope_index_pages import (
    gen_main_index_page,
    gen_model_page,
    get_model_config,
    make_page_file,
    make_random_redirect_2d,
)

from neuroscope.scan_over_data import (
    scan_over_data
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
parser.add_argument(
    '--max_tokens', type=int, default=-1,
    help="Max number of tokens to run through neuroscope. -1 for entire dataset"
)

args = parser.parse_args()

if args.model:
    model_name = args.model
    print(f"Building for {model_name}")
    main_index_html = gen_main_index_page([model_name])
    main_index_path = os.path.join(WEBSITE_DIR, "index.html")
    make_page_file(main_index_path, main_index_html)
    print(f"Wrote {main_index_path}")
    cfg = get_model_config(model_name)
    model_html = gen_model_page(model_name)
    make_page_file(os.path.join(WEBSITE_DIR, model_name, "index.html"), model_html)
    make_page_file(os.path.join(WEBSITE_DIR, model_name, "model.html"), REDIRECT_TO_INDEX)
    make_page_file(os.path.join(WEBSITE_DIR, model_name, "random.html"), make_random_redirect_2d(cfg.n_layers, cfg.d_mlp))
    print(f"Wrote {WEBSITE_DIR}/(index|model|random).html")
    trackers = scan_over_data(max_tokens=args.max_tokens)
    print('Finished')


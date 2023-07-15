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
    cfg = get_model_config(model_name)
    model_html = gen_model_page(model_name)
    make_page_file(os.path.join(folder, "index.html"), model_html)
    make_page_file(os.path.join(folder, "model.html"), REDIRECT_TO_INDEX)
    make_page_file(os.path.join(folder, "random.html"), make_random_redirect_2d(cfg.n_layers, cfg.d_mlp))
    print(f"Wrote {folder}/(index|model|random).html")

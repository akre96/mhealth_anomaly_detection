import json
from typing import Dict


def get_colors(path: str = "lib/colors.json") -> Dict:
    with open(path, "r") as p:
        return json.load(p)


def get_all_feature_params(path: str = "lib/feature_parameters.json") -> Dict:
    with open(path, "r") as p:
        return json.load(p)

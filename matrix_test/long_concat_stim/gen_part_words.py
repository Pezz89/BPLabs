#!/usr/bin/env python3
import sys
sys.path.insert(0, "../helper_modules/")

import csv
from tokens_to_words import tokens_to_words, load_component_map
import pdb

def main():
    component_map_file = "../speech_components/component_map.json"
    stim_parts = "./out/parts/stim_parts.csv"
    stim_words = "./out/parts/stim_words.csv"
    component_map = load_component_map(component_map_file)
    lines = []
    with open(stim_parts, 'r') as csvfile:
        for line in csv.reader(csvfile):
            lines.append(tokens_to_words(line, component_map))
    with open(stim_words, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(lines)


if __name__ == "__main__":
    main()

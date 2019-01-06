#!/usr/bin/env bash
cd ./behavioural_stim
./generate_behavioural_stimulus.sh
cd ../short_concat_stim
./short_concat_stim.py
cd ../long_concat_stim
./generate_long_deconder_stimulus.sh

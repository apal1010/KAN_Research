#!/bin/bash

python3 benchmark_custom.py --output-path fb_results  \
--method all \
--batch-size 32 --inp-size 1000 --hid-size 1000 --reps 5
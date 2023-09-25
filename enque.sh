#!/bin/bash

for g in $(cat graphs)
do
  for i in $(seq 0 9)
  do
    scrapyd-client schedule -p gd low_rank_full_sgd --arg graph=$g --arg iterations=15 --arg eps=0.1 --arg unit_edge_length=1 --arg p=${i}0 --arg seed_start=0 --arg seed_stop=100
    scrapyd-client schedule -p gd distance_adjusted_full_sgd --arg graph=$g --arg iterations=15 --arg eps=0.1 --arg unit_edge_length=1 --arg l=0.${i} --arg seed_start=0 --arg seed_stop=100
    scrapyd-client schedule -p gd distance_adjusted_sparse_sgd --arg graph=$g --arg iterations=15 --arg eps=0.1 --arg unit_edge_length=1 --arg l=0.${i} --arg pivot=200 --arg seed_start=0 --arg seed_stop=100
  done
done
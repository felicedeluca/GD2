#!/bin/bash

for file in "/Users/felicedeluca/Downloads/SPX-graph-layout-master/small_27_dags_iterations_10_4_spx"/*.dot
do
  echo "$file"
  python3 GD2Main_planar.py "$file"
done

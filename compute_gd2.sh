#!/bin/bash

for file in "/Users/felicedeluca/Downloads/small_27_dags_iterations_10_4_spx"/*.dot
do
  echo "$file"
  python3 GD2Main_planar.py "$file"
done

for file in "/Users/felicedeluca/Developer/GD2/output"/*.dot
do
  echo "$file"
  neato  -n2 -Nheight=0.01 -Nwidth=0.01 -Nfixedsize=true -Nlabel="" -Earrowsize=0.1 "$file" -Tpdf > "${file}.pdf"
done

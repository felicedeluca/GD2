#!/bin/bash

for file in "/Users/felicedeluca/Downloads/output_layouts_wgt_stress"/*.dot
do
  echo "$file"
  neato  -n2 -Nheight=0.01 -Nwidth=0.01 -Nfixedsize=true -Nlabel="" -Earrowsize=0.1 "$file" -Tpdf > "${file}.pdf"
done

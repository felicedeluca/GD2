#!/bin/bash

for file in "/Users/felicedeluca/Developer/GD2/output"/*.dot
do
  echo "$file"
  neato  -n2 "$file" -Tpdf > "${file}.pdf"
done

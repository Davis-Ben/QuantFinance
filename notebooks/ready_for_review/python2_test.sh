#!/bin/bash
for filename in *.ipynb; do
	echo $filename
    jupyter nbconvert --ExecutePreprocessor.kernel_name=python2 --execute --ExecutePreprocessor.timeout=600 "$filename" || echo "$filename" >> errors.txt
done
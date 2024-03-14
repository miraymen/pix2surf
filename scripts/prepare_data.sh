#!/usr/bin/env bash

wget -O pix2surf.zip https://hpsdata.mpi-inf.mpg.de/pix2surf.zip
unzip pix2surf.zip
rm pix2surf.zip
mv data ./train/

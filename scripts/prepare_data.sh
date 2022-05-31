#!/usr/bin/env bash

wget https://nextcloud.mpi-klsb.mpg.de/index.php/s/ozRcGdGwAJ3tBns
unzip pix2surf.zip
rm pix2surf.zip
mv data ./train/

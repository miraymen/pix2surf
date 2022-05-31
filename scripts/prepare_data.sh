#!/usr/bin/env bash

wget -O pix2surf.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/ozRcGdGwAJ3tBns/download
unzip pix2surf.zip
rm pix2surf.zip
mv data ./train/

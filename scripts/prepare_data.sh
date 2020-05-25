#!/usr/bin/env bash

gdown https://drive.google.com/uc?id=1ULtdEXRrxH9_CtTrWensIbwybeWKz8Dj
unzip pix2surf.zip
rm pix2surf.zip
mv data ./train/

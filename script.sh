#!/bin/sh

python make_data.py

( cd mimicAudio && docker-compose up )
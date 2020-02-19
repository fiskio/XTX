#!/usr/bin/env bash

docker run --rm -it -v `pwd`:/XTX -w /XTX -p 8888:8888 fiskio/ml-base $@

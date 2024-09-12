#!/bin/bash
git submodule foreach --recursive git clean -xfd
git submodule foreach --recursive git reset --hard
git submodule update --init --recursive
rm -rf ../MadGraph5-v3.5.5/*

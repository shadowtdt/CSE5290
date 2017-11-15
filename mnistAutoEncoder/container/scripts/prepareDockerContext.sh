#!/bin/bash

TARGET_DIR=target
BOLD="\033[1m"
NORM="\033[0m"

# Destroy old python files
if [ -d "${TARGET_DIR}" ]; then
    echo -e "${BOLD}** Destroying old python files **${NORM}"
    rm -r ${TARGET_DIR}
fi

echo -e "${BOLD}** Copying python files into docker context **${NORM}"
mkdir ${TARGET_DIR}
cp ../*.py ${TARGET_DIR}
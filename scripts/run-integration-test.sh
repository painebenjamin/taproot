#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DIR=$(realpath ${SCRIPT_DIR}/..)
cd ${DIR}/src/

export CUDA_MODULE_LOADING=LAZY
export KMP_DUPLICATE_LIB_OK=TRUE

if [[ $1 =~ ^[0-9]+$ ]]; then
    TEST=$(find . -type f -name "test_$1*.py")
    # Replace `/` with `.` and remove `.py`
    TEST=$(echo $TEST | sed 's/\.py//g' | sed 's/\.//g' | sed 's/^\///g' | sed 's/\//\./g')
    echo "Running test: $TEST"
    shift
    python -m pytest --pyargs $TEST $@
else
    echo "Running all tests"
    python -m pytest --pyargs taproot.tests $@
fi

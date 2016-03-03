#!/usr/bin/bash

all_tests_passed=true

for test in `find . -name "test_*" -type f`; do
    if ! bash $test; then
        all_tests_passed=false
    fi
done

if [ "$all_tests_passed" = true ]; then echo "ALL TESTS PASSED!"; fi

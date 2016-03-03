#!/usr/bin/bash

function rmv_trailing_empty_lines {
    sed -e :a -e '/^\n*$/{$d;N;};/\n$/ba'
}

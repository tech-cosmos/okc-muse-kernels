#!/bin/bash
. ~/okc/bin/activate
d=$(dirname $1); m=$(basename $1 .py)
so=$(python -c "import sys; sys.path.insert(0,'$d'); import $m as S; print(S._lib().__file__)" 2>/dev/null | tail -1)
C=/usr/local/cuda-13.0/bin/cuobjdump; echo $so
$C -res-usage $so 2>&1 | grep -A1 "moe_twm" | grep -o "REG:[0-9]*\|STACK:[0-9]*" | paste - - | head -1
$C -sass $so 2>/dev/null | awk '/Function : /{f=/moe_twm/} f' | grep -c "LDL\|STL"

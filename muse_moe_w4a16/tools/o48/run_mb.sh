#!/bin/bash
# o48: o47 + o46-best microbenches in one go (cwd = muse_moe_w4a16). Used on the Modal H200 via h200m.py and on the H100.
export CUDA_HOME=${CUDA_HOME:-$( [ -d /usr/local/cuda-13.0 ] && echo /usr/local/cuda-13.0 || echo /usr/local/cuda )}
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader; $CUDA_HOME/bin/nvcc --version | tail -2
python -c "import torch; print('torch', torch.__version__)"
echo "## SASS (mb_f16p, this build): key regs/instr/alu/HGMMA/HMUL2 (ty 0-2: dq-only, both; ty 3-8 both)"
SASSONLY=1 python tools/o48/mb_f16p.py 2>&1
echo "## mb_f16p grid 132"
python tools/o48/mb_f16p.py 132 2>&1 | grep -v "^PTXAS\|^$"
echo "## mb_mmasync16 grid 132"
python tools/o47/mb_mmasync16.py 132 2>&1
echo "## mb_chain grid 132 (CH1 base, CH8 chain WD1 N64, CH4 chain WD0 N32)"
CHV="2,64,1,0,1:2,64,8,1,1:4,32,1,0,1:4,32,4,1,0" python tools/o45/mb_chain.py 132 2>&1

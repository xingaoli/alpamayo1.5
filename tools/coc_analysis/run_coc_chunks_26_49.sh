#!/bin/bash

# Run CoC batch inference on chunks 26-49
python3 /home/xingao/code/Alpamayo1.5/tools/coc_analysis/1_batch_coc_inference.py --chunks \
  chunk_0026 chunk_0027 chunk_0028 chunk_0029 chunk_0030 \
  chunk_0031 chunk_0032 chunk_0033 chunk_0034 chunk_0035 \
  chunk_0036 chunk_0037 chunk_0038 chunk_0039 chunk_0040 \
  chunk_0041 chunk_0042 chunk_0043 chunk_0044 chunk_0045 \
  chunk_0046 chunk_0047 chunk_0048 chunk_0049

#!/bin/bash

# Run CoC batch inference on chunks 0-25
python3 /home/xingao/code/Alpamayo1.5/tools/coc_analysis/1_batch_coc_inference.py --chunks \
  chunk_0000 chunk_0001 chunk_0002 chunk_0003 chunk_0004 \
  chunk_0005 chunk_0006 chunk_0007 chunk_0008 chunk_0009 \
  chunk_0010 chunk_0011 chunk_0012 chunk_0013 chunk_0014 \
  chunk_0015 chunk_0016 chunk_0017 chunk_0018 chunk_0019 \
  chunk_0020 chunk_0021 chunk_0022 chunk_0023 chunk_0024 \
  chunk_0025

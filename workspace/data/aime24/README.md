---
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: url
    dtype: string
  splits:
  - name: test
    num_bytes: 13290
    num_examples: 30
  download_size: 11183
  dataset_size: 13290
configs:
- config_name: default
  data_files:
  - split: test
    path: test-*
---

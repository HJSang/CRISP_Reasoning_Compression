---
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: answer
    dtype: string
  - name: id
    dtype: string
  splits:
  - name: test
    num_examples: 30
configs:
- config_name: default
  data_files:
  - split: test
    path: test.jsonl
---

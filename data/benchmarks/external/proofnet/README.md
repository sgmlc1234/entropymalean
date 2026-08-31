---
dataset_info:
  features:
  - name: name
    dtype: string
  - name: split
    dtype: string
  - name: informal_prefix
    dtype: string
  - name: formal_statement
    dtype: string
  - name: goal
    dtype: string
  - name: header
    dtype: string
  - name: formal_proof
    dtype: string
  splits:
  - name: test
    num_bytes: 106803
    num_examples: 186
  - name: valid
    num_bytes: 106390
    num_examples: 185
  - name: few_shot_examples
    num_bytes: 52461
    num_examples: 67
  download_size: 128962
  dataset_size: 265654
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: valid
    path: data/valid-*
  - split: few_shot_examples
    path: data/few_shot_examples-*
---

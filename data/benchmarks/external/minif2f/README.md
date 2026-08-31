---
annotations_creators:
  - expert-generated
language_creators:
  - expert-generated
language:
  - en
license:
  - mit
multilinguality:
  - monolingual
size_categories:
  - 1K<n<10K
source_datasets:
  - original
task_categories:
  - text-generation
  - other
task_ids:
  - explanation-generation
  - language-modeling
paperswithcode_id: minif2f
pretty_name: minif2f
tags:
  - formal-mathematics
  - theorem-proving
  - mathematical-reasoning
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
  config_name: default
---

# minif2f Dataset

The minif2f dataset is a collection of mathematical problems and their formal statements, designed for formal mathematics and theorem proving tasks.

## Dataset Description

### Dataset Summary

The minif2f dataset contains mathematical problems from various sources (like AMC competitions) along with their formal statements in the Lean theorem prover format. Each example includes both informal mathematical statements and their corresponding formal representations.

### Supported Tasks

The dataset supports the following tasks:
- Formal Mathematics
- Theorem Proving
- Mathematical Reasoning
- Translation between informal and formal mathematics

### Languages

The dataset contains:
- Natural language mathematical statements (English)
- Formal mathematical statements (Lean theorem prover syntax)

## Dataset Structure

### Data Instances

Each instance in the dataset contains the following fields:

```python
{
    'name': str,           # Unique identifier for the problem
    'split': str,          # Dataset split (train/valid/test)
    'informal_prefix': str, # Informal mathematical statement in LaTeX
    'formal_statement': str, # Formal statement in Lean syntax
    'goal': str,           # The goal state to be proved
    'header': str          # Import statements and configuration
}
```

Example:
```python
{
    "name": "amc12a_2015_p10",
    "split": "valid",
    "informal_prefix": "/-- Integers $x$ and $y$ with $x>y>0$ satisfy $x+y+xy=80$. What is $x$? ...",
    "formal_statement": "theorem amc12a_2015_p10 (x y : ℤ) (h₀ : 0 < y) (h₁ : y < x) (h₂ : x + y + x * y = 80) : x = 26 := by\n",
    "goal": "x y : ℤ\nh₀ : 0 < y\nh₁ : y < x\nh₂ : x + y + x * y = 80\n⊢ x = 26",
    "header": "import Mathlib\nimport Aesop\n..."
}
```

### Data Splits

The dataset is divided into the following splits:
- train
- valid
- test

### Data Fields

- `name`: Unique identifier for the mathematical problem
- `split`: Indicates which split the example belongs to
- `informal_prefix`: The informal mathematical statement in LaTeX format
- `formal_statement`: The formal theorem statement in Lean theorem prover syntax
- `goal`: The goal state that needs to be proved
- `header`: Required imports and configuration for the formal statement

## Dataset Creation

### Source Data

The problems in this dataset are sourced from various mathematical competitions and problems, including the American Mathematics Competition (AMC).

### Annotations

The formal statements are manually created by experts in formal mathematics, translating informal mathematical problems into the Lean theorem prover format.

## Additional Information

### License

mit

### Citation

@inproceedings{
  2210.12283,
  title={Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs},
  author={Albert Q. Jiang and Sean Welleck and Jin Peng Zhou and Wenda Li and Jiacheng Liu and Mateja Jamnik and Timothée Lacroix and Yuhuai Wu and Guillaume Lample},
  booktitle={Submitted to The Eleventh International Conference on Learning Representations},
  year={2022},
  url={https://arxiv.org/abs/2210.12283}
}

@article{lin2024Goedelprover,
      title={Goedel-Prover: A New Frontier in Automated Theorem Proving}, 
      author={Yong Lin and Shange Tang and Bohan Lyu and Jiayun Wu and Hongzhou Lin and Kaiyu Yang and Jia Li and Mengzhou Xia and Danqi Chen and Sanjeev Arora and Chi Jin},
}


### Contributions

Contributions to the dataset are welcome. Please submit a pull request or open an issue to discuss potential changes or additions.

## Using the Dataset

You can load the dataset using the Hugging Face datasets library:

```python
from datasets import load_dataset

dataset = load_dataset("minif2f")
```

## Contact

use community tab
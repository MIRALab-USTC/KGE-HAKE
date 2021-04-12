# HAKE: Hierarchy-Aware Knowledge Graph Embedding
This is the code of paper **Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction.** *Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, Jie Wang.* AAAI 2020.  [arxiv](https://arxiv.org/abs/1911.09419)

## Dependencies
- Python 3.6+
- [PyTorch](http://pytorch.org/) 1.0+

## Results
The results of **HAKE** and the baseline model **ModE** on **WN18RR**, **FB15k-237** and **YAGO3-10** are as follows.
 
### WN18RR
| | MRR |  HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ModE | 0.472 | 0.427 | 0.486 | 0.564 |
| HAKE | 0.496 ± 0.001 | 0.452 | 0.516 | 0.582 |


### FB15k-237
| | MRR | HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ModE | 0.341 |  0.244 | 0.380 | 0.534 |
| HAKE | 0.346 ± 0.001 |  0.250 | 0.381 | 0.542 |

### YAGO3-10
| | MRR | HITS@1 | HITS@3 | HITS@10 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ModE | 0.510 |  0.421 | 0.562 | 0.660 |
| HAKE | 0.546  ± 0.001 |  0.462 | 0.596 | 0.694 |


## Running the code 

### Usage
```
bash runs.sh {train | valid | test} {ModE | HAKE} {wn18rr | FB15k-237 | YAGO3-10} <gpu_id> \
<save_id> <train_batch_size> <negative_sample_size> <hidden_dim> <gamma> <alpha> \
<learning_rate> <num_train_steps> <test_batch_size> [modulus_weight] [phase_weight]
```
- `{ | }`: Mutually exclusive items. Choose one from them.
- `< >`: Placeholder for which you must supply a value.
- `[ ]`: Optional items.

**Remark**: `[modulus_weight]` and `[phase_weight]` are available only for the `HAKE` model.

To reproduce the results of HAKE and ModE, run the following commands.

### HAKE
```
# WN18RR
bash runs.sh train HAKE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 0.5 0.5

# FB15k-237
bash runs.sh train HAKE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 3.5 1.0

# YAGO3-10
bash runs.sh train HAKE YAGO3-10 0 0 1024 256 500 24.0 1.0 0.0002 180000 4 1.0 0.5
```

### ModE
```
# WN18RR
bash runs.sh train ModE wn18rr 0 0 512 1024 500 6.0 0.5 0.0001 80000 8 --no_decay

# FB15k-237
bash runs.sh train ModE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.0001 100000 16

# YAGO3-10
bash runs.sh train ModE YAGO3-10 0 0 1024 256 500 24.0 1.0 0.0002 80000 4
```

## Visualization
To plot entity embeddings on a 2D plane (Figure 4 in our paper), please refer to this [issue](https://github.com/MIRALab-USTC/KGE-HAKE/issues/2).

## Citation
If you find this code useful, please consider citing the following paper.
```
@inproceedings{zhang2020learning,
  title={Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction},
  author={Zhang, Zhanqiu and Cai, Jianyu and Zhang, Yongdong and Wang, Jie},
  booktitle={Thirty-Fourth {AAAI} Conference on Artificial Intelligence},
  pages={3065--3072},
  publisher={{AAAI} Press},
  year={2020}
}
```

## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). Thanks for their contributions.

## Other Repositories
If you are interested in our work, you may find the following paper useful.

**Duality-Induced Regularizer for Tensor Factorization Based Knowledge Graph Completion.**
*Zhanqiu Zhang, Jianyu Cai, Jie Wang.* NeurIPS 2020. [[paper](https://arxiv.org/abs/2011.05816)] [[code](https://github.com/MIRALab-USTC/KGE-DURA)]


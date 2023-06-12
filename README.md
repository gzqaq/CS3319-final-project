# Final Project of CS3319 Foundation of Data Science

Code of the final project of CS3319 Foundation of Data Science, Spring 2023, SJTU

## Environment setup

```terminal
conda env create -f env.yml
conda activate gnn-rec
```

## Reproduction

### Train

```terminal
python train.py --n_epochs 3000 --lr 0.001 --model puregcn --gpu <gpu> --run_name <dir_name>
```

### Evaluation

```terminal
python eval.py --score <f1_score_when_saved> --save_name <dir_name>
```

### Experiments for further analysis

``terminal
python experiment.py --lr <lr> --nbh <nbh_size> --loss <loss_type> --model <model_name> --gpu <gpu> --run_name <dir_name>
```

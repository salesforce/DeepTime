# DeepTIMe: Deep Time-Index Meta-Learning for Non-Stationary Time-Series Forecasting

<p align="center">
<img src=".\pics\deeptime.png" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall approach of DeepTIMe.
</p>

Official PyTorch code repository for the [DeepTIMe paper](https://arxiv.org/abs/2207.06046).

* DeepTIMe is a deep time-index based model trained via a meta-learning formulation, yielding a strong method for
  non-stationary time-series forecasting.
* Experiments on real world datases in the long sequence time-series forecasting setting demonstrates that DeepTIMe
  achieves competitive results with state-of-the-art methods and is highly efficient.

## Requirements

Dependencies for this project can be installed by:

```bash
pip install -r requirements.txt
```

## Quick Start

### Data

To get started, you will need to download the datasets as described in our paper:

* Pre-processed datasets can be downloaded from the following
  links, [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/)
  or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing), as obtained
  from [Autoformer's](https://github.com/thuml/Autoformer) GitHub repository.
* Place the downloaded datasets into the `storage/datasets/` folder, e.g. `storage/datasets/ETT-small/ETTm2.csv`.

### Reproducing Experiment Results

We provide some scripts to quickly reproduce the results reported in our paper. There are two options, to run the full
hyperparameter search, or to directly run the experiments with hyperparameters provided in the configuration files.

__Option A__: Run the full hyperparameter search.

1. Run the following command to generate the experiments: `make build-all path=experiments/configs/hp_search`.
2. Run the following script to perform training and evaluation: `./run_hp_search.sh` (you may need to
   run `chmod u+x run_hp_search.sh` first).

__Option B__: Directly run the experiments with hyperparameters provided in the configuration files.

1. Run the following command to generate the experiments: `make build-all path=experiments/configs`.
2. Run the following script to perform training and evaluation: `./run.sh` (you may need to run `chmod u+x run.sh`
   first).

Finally, results can be viewed on tensorboard by running `tensorboard --logdir storage/experiments/`, or in
the `storage/experiments/experiment_name/metrics.npy` file.

## Main Results

We conduct extensive experiments on both synthetic and real world datasets, showing that DeepTIMe has extremely
competitive performance, achieving state-of-the-art results on 20 out of 24 settings for the multivariate forecasting
benchmark based on MSE.
<p align="center">
<img src=".\pics\results.png" width = "700" alt="" align=center />
<br><br>
</p>

## Detailed Usage

Further details of the code repository can be found here. The codebase is structured to generate experiments from
a `.gin` configuration file based on the `build.variables_dict` argument.

1. First, build the experiment from a config file. We provide 2 ways to build an experiment.
    1. Build a single config file:
       ```
       make build config=experiments/configs/folder_name/file_name.gin
       ```
    2. Build a group of config files:
       ```bash
       make build-all path=experiments/configs/folder_name
       ```
2. Next, run the experiment using the following command
    ```bash 
    python -m experiments.forecast --config_path=storage/experiments/experiment_name/config.gin run
   ```
   Alternatively, the first step generates a command file found in `storage/experiments/experiment_name/command`, which
   you can use by the following command,
   ```bash
   make run command=storage/experiments/experiment_name/command
   ```
3. Finally, you can observe the results on tensorboard
   ```bash
   tensorboard --logdir storage/experiments/
   ``` 
   or view the `storage/experiments/deeptime/experiment_name/metrics.npy` file.

## Acknowledgements

The implementation of DeepTIMe relies on resources from the following codebases and repositories, we thank the original
authors for open-sourcing their work.

* https://github.com/ElementAI/N-BEATS
* https://github.com/zhouhaoyi/Informer2020
* https://github.com/thuml/Autoformer

## Citation

Please consider citing if you find this code useful to your research.
<pre>@article{woo2022deeptime,
    title={DeepTIMe: Deep Time-Index Meta-Learning for Non-Stationary Time-Series Forecasting},
    author={Gerald Woo and Chenghao Liu and Doyen Sahoo and Akshat Kumar and Steven C. H. Hoi},
    year={2022},
    url={https://arxiv.org/abs/2207.06046},
}</pre>
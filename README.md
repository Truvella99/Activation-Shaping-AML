# Aml_project

Project of the Advanced Machine Learning course, made in <a href='https://www.python.org/' target='_blank'>python</a> with the <a href='https://code.visualstudio.com/' target='_blank'>VsCode</a> IDE. 
<br>The goal of the project was to address the domain shift problem, where the accuracy of the model significantly drops when other domain with different data distribution were involved. For this reason, techniques like UDA (unsupervised domain adaptation) have been used and implemented (by leveraging on the Activation Shaping Hook) in order to cope with this problem. <br>The project was made with the <a href='https://pytorch.org/' target='_blank'>Pytorch</a> framework, using <a href='https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html' target='_blank'>resnet18</a> with pretrained weights as base architecture, and both the <a href='https://code.visualstudio.com/' target='_blank'>VsCode</a> IDE and <a href='https://colab.google/' target='_blank'>Google Colab</a> as runtime environments.<br><br>
More details can be found [here](paper.pdf).

# AML 2023/2024 Project - Activation Shaping for Domain Adaptation
Official repository for the "Activation Shaping for Domain Adaptation" project - Advanced Machine Learning Course 2023/2024 @ PoliTo

## Getting Started
Make sure to have a CUDA capable device, supporting at least CUDA 11.4, installed and correctly configured on your system. 

(The base code of this project has been produced using CUDA 11.4 and Python 3.10.9)

Once you have properly setup everything, make sure you are in the correct directory and run from the command line:
```bash
pip install -r requirements.txt
```

### Dataset
1. Download the PACS dataset from the portal of the course in the "project_topics" folder.
2. Place the dataset in the 'data/PACS' folder making sure that the images are organized in this way:
```
data/PACS/kfold/art_painting/dog/pic_001.jpg
data/PACS/kfold/art_painting/dog/pic_002.jpg
data/PACS/kfold/art_painting/dog/pic_003.jpg
...
```

At this point you should be able to run and edit the base code provided.

*NOTE: please, do not push the PACS dataset into your github repository.*

## Base Code Structure
The starting code should already provide everything needed to easily extend it. Specifically, you should be able to implement all the points of the project by 
editing only the code in the dedicated sections of each file (denoted by '##..#' separators and '#TODO'). However, feel free to proceed with the project in the way that best suits your needs.

In the following, you can find a brief description of the included files, alongside where it is suggested to edit the code.

| File | Description | Code to Implement |
| ---- | ----------- | ----------------- |
| `main.py` | main entry point. Contains the logic needed to properly setup and run each experiment. | (1) Model loading for each experiment. (2) Training logic for each experiment. |
| `parse_args.py` | contains the function responsible for parsing each command line argument. | - |
| `globals.py` | contains the global variables of the program. | - |
| `dataset/PACS.py` | contains the code to load data, build splits and dataloaders. | Loading and Dataset object creation for each experiment. |
| `dataset/utils.py` | contains utilities (eg. datasets classes) to correctly setup the data pipeline. | Dataset classes for each experiment. |
| `models/resnet.py` | contains the architectures and modules used in the project. | (1) Activation Shaping Module. (2) Custom ResNet18 with the Activation Shaping Module. |

## Running The Experiments
To run the experiments you can use, copy and modify the provided launch script `launch_scripts/baseline.sh`.
As an example, to reproduce the baseline you can launch the three experiments as
```
./launch_scripts/baseline.sh cartoon
./launch_scripts/baseline.sh sketch
./launch_scripts/baseline.sh photo
```
where the argument following the invocation of the script is programmed to be the target domain.

*NOTE: you should upload with your code also these scripts as they are fundamental to run your code and reproduce your results.*

In the following, you can find a brief description of the relevant command line arguments when launching the experiments.

### Base Command Line Arguments
| Argument &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  | Description |
| -------- | ----------- |
| `--experiment` | the identifier of the experiment to run. (Eg. `baseline`) |
| `--experiment_name` | the name of the path where to save the checkpoint and logs of the experiment. (Eg. `baseline/cartoon`) |
| `--experiment_args` | a string that gets converted at runtime in a python dict, containing the experiment's specifications. |
| `--dataset_args` | a string that gets converted at runtime in a python dict, containing the dataset's specifications (eg. the root folder of the dataset, the source domain, the target domain...) |
| `--batch_size` | batch size used in the optimization procedure. The default value should be fine for GPUs with at least 4GB of dedicated GPU memory. *Do not change it*, unless you have to reduce batch size. In that case, you can emulate larger batch sizes by tuning the `grad_accum_steps` argument accordingly. (Eg. `(--batch_size=128 --grad_accum_steps=1) OR (--batch_size=64 --grad_accum_steps=2)`) |
| `--epochs` | total number of epochs of the optimization procedure. *Do not change it*, it defaults to 30 epochs. |
| `--num_workers` | total number of worker processes to spawn to speed up data loading. Tweak it according to your hardware specs. |
| `--grad_accum_steps` | controls how many forward-backward passes to accumulate before updating the parameters. It can be used to emulate larger batch sizes if needed, see the `batch_size` argument above. |
| `--cpu` | if set, the experiment will run on the CPU. |
| `--test_only` | if set, the experiment will skip the training procedure and just run the evaluation on the test set. |
| `--seed` | the integer used to seed all the experiments, to make results as reproducible as possible. *Do not change it*, it defaults to 0. |

## Baseline Results (see point 0. of the project)
|          | Art Painting &#8594; Cartoon | Art Painting &#8594; Sketch | Art Painting &#8594; Photo | Average |
| :------: | :--------------------------: | :-------------------------: | :------------------------: | :-----: |
| Baseline |            54.52             |             40.44           |            95.93           |  63.63  |


## Bug Reporting
You are encouraged to share in the project's dedicated Slack channel any bugs you discover in the code. It's incredibly valuable for everyone involved to be aware of any issues as soon as they arise, helping to address them promptly. Your contribution is greatly appreciated!

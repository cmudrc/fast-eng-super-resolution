# fast-eng-super-resolution
Fast super-resolution for engineering simulations on 3D unstructured meshes

## Introduction
This repository contains the code for the paper "Fast super-resolution analysis of low-pressure duct air flow through adaptive domain decomposition" by [Wenzhuo Xu](https://wenzhuoxu.com/), [Akibi Archer](https://www.akibiarcher.com/), Mike McCarrell, Scott Hesser, [Noelia Grande Gutiérrez](https://www.meche.engineering.cmu.edu/directory/bios/grande-gutierrez-noelia.html) and [Christopher McComb](https://www.meche.engineering.cmu.edu/directory/bios/mccomb-christopher.html). 

## Quick Start
Use the following command to configure the environment:
```bash
pip install -r requirements.txt
```

Training and prediction on the framework can be done by running the python script in the project folder. An example using the duct geometry dataset, and neural operator (NO) is shown below:
```bash
python run_DS_3D.py --dataset=duct --model=neuralop --mode=train
```

After excuting the above command, the model will be trained on the duct dataset. To predict the results, run the following command:
```bash
python run_DS_3D.py --dataset=duct --model=neuralop --mode=predict
```

All results will be saved to the `./logs` folder.
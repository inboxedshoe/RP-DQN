# Routing Problem Deep Q-Network (RP-DQN)
This is the codebase of the paper "RP-DQN: An application of Q-Learning to Vehicle Routing Problems", submitted 25 April 2021.  

In this paper we present a new approach to tackle complex
routing problems with an improved state representation that utilizes the
model complexity better than previous methods. We enable this by training from temporal differences. Specifically Q-Learning is employed. We
show that our approach achieves state-of-the-art performance for autoregressive policies that sequentially insert nodes to construct solutions on
the CVRP. Additionally, we are the first to tackle the MDVRP with machine learning methods and demonstrate that this problem type greatly
benefits from our approach over other ML methods.

## Paper
If you use our code, please cite as:
```
@artice{rpdqn2021,
title = {},
author = {Ahmad Bdeir and Simon Boeder and Tim Dernedde and Kirill Tkachuk},
year = {2021}
}

```

## Dependencies
- Python 3.x
- NumPy
- PyTorch
- tensorboard


## GUI Dependencies
- pyside2 (install through conda-forge)
- pyqt5


## Usage

Put here how to:
- Model Training:
	- Edit the run settings, in the main.py file at project root
	- Run the main.py file
	
- Running a pretrained model through GUI:
	- Run the main-qt.py file in the GUI folder
	- Select The parent model folder containing the model subfolders
	- Select the model subfolder through drop down and the model from list
	- Select the validation test file
	- Test the model
	- Sampling is also straight forward in first tab
	
	Note: After the third step, the trained model settings become visible in the "config" tab
	
- Running a pretrained model through:
	- An example is found in the Evaluate_Example file at root.

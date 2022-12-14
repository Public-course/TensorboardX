# TensorboardX (python 3)
Using tensorboardX (tensorboard for pytorch) e.g. ploting more than one graph in the same chat or Netwoks Graph and etc.


### Install

1. Install PyTorch: https://pytorch.org/get-started/locally/
2. Install TensorboardX: ```pip install tensorboardX``` (https://tensorboardx.readthedocs.io/en/latest/tutorial.html#install)

### Run the code
Clone the repository:
```diff
git clone https://github.com/Public-course/TensorboardX.git
```
and open it:
```diff
cd TensorboardX
```

##### visualization of plotted scalar values:
1. Run the script, e.g.:<br>
.../TensorboardX$ ```python more_plots_one_chat.py ```
2. Run TensorBoardX (in another Terminal):<br>
.../TensorboardX$  ```tensorboard --logdir runs```
3. See plot in the browser (tensorboard print out a link for visualization in step 2)

##### visualization of Network as Graph:
1. .../TensorboardX$ ```python plot_graph_net.py```
2. .../TensorboardX$ ```tensorboard --logdir runs```
3. See the graph in the browser

<p align="center">
  <img src="https://github.com/Public-course/TensorboardX/blob/master/tensorboard.png" />

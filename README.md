# DRL project
## It's better than it looks, I promise
This might look a bit like a mess but it isn't so bad (not really). The **main** code is effectively imported in the 'main.py'. Most of this code is hopefully quite readable, but if not, send me a message and I'll happily explain. Some neural networks are constructed in separate .py-files.

A few of the plots we used are saved as separate pngs. Not all were used in the final project, and can safely be ignored (but they are pretty to look at, so take a look if you want to).

Four jupyter notebooks contain the code that generated and visualized relevant data for the project. These are inteded to be a complement to the report, though they are not necessary to read to understand the report:

  1. 'Training experiments.ipynb' which shows the training of a 'local agent'.
  2. 'rwnet variants.ipynb' shows some tuning of the best performing 'local agent'.
  3. 'Softmax experiments.ipynb' shows the training and deployment of a 'local agent' using a softmax policy. This also contains some qualitative visualization. We invite the interested reader to try it out, its quite fun to see what the agent gets up to and how it assesses the world.
  4. 'global.ipynb' shows the (arguably failed) attempt at making a 'local agent' a 'global agent' by adjusting its local map based on the full map. It also shows generalizability of agents, and again shows some qualitative data.


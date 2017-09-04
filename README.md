# OpenAI-CartPole-v0

<div align="center">
  <img src="https://media.giphy.com/media/iNplDboNzHXk4/giphy.gif"><br><br>
</div>

OpenAI's CartPole-v0 environment is described as follows: a pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center. CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.

To solve this problem I combined convolutional, long short-term memory and fully connected networks to form a CLDNN as described in Google's [research paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43455.pdf). The neural network was implemented using TFLearn, and this is the architecture of the neural network:

<div align="center">
  <br><img src="https://cldup.com/FMIGBVPy8q.png" width="400" height="233.4"><br><br>
</div>

The evaluation of my algorithm can be found [here](https://gym.openai.com/evaluations/eval_qZJDndcBSXqB0t3hBxJVVw). 
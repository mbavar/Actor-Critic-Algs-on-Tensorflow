## A3C-vs-A2C-on-tensorflow

The goal of this project is to provide high quality implementations of a few of popular distributed Reinforcement Learning methods. In the first phase, we focus on the basic Actor-Critic and its asynchronous version A3C. We later plan to implement and compare these with A2C (Synchronous Actor Critic)  in a few OpenAI gym environments in terms of time and sample complexities. 

### Requirements: 
OpenAI gym, Python3, Tensorflow 1.3.

See [Tensorflow](https://www.tensorflow.org/install/) and [gym](https://gym.openai.com/docs/) installation pages for specific details.

We recommend using a package manager such as conda for convenience. 

### Basic Actor Critic
To run this on an enviroment such as `CartPole-v0` simply run
```
cd Basic_AC
python run_AC.py --env CartPole-v0 --animate
```
The default environment without specifying `--env` option is `Pendulum-v0`.


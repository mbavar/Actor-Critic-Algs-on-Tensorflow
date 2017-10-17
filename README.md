## A3C-vs-A2C-on-tensorflow

The goal of this project is to provide high quality implementations of a few popular distributed Reinforcement Learning methods. In the first phase, we focus on the basic Actor-Critic and its more sophisticated variant A3C. We also test these implementations on a few OpenAI gym environments. 

### Requirements
OpenAI gym, Python3, Tensorflow 1.3.

See [Tensorflow](https://www.tensorflow.org/install/) and [gym](https://gym.openai.com/docs/) installation pages for specific details.

We recommend using a package manager such as conda for convenience. 

### Basic Actor Critic
To run this on an enviroment such as `CartPole-v0` simply run
```
cd Basic_AC
python run_AC.py --env CartPole-v0 --animate
```
The default environment without specifying `--env` flag is `Pendulum-v0`.

### A3C

The A3C algorithm was introduced by researchers from Google DeepMind [Mnih et al.](https://arxiv.org/abs/1602.01783) as a way to extend the popular Actor-Critic method to distributed setting. Most implementations of A3C in Tensorflow, with the exception of OpenAI [univese starter agent](https://github.com/openai/universe-starter-agent), currently use threading. In our case, we use the native [distributed Tensorflow](https://www.tensorflow.org/deploy/distributed) capabilities, which has the advantage of being more efficient when deploying a large number of workers.

To run a basic version of our A3C implementation
```
cd A3C
bash runner.sh
```
The bash file `runner.sh` launches 3 processes on the localhost with one `ps` (parameter server) job and two workers. To increase the number of workers or servers, edit the appropriate variables in `runner.sh`.



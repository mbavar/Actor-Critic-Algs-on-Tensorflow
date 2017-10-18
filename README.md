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

### Testing the models

The chief process has the responsibility of saving the model. This is done every 600 epochs by default. The frequency and the location of the checkpoints can be  controlled via `--save_every` and `--checkpoint_dir` flags. You can test the saved models using test_model.py , e.g.
```
cd A3C
python test_model.py Pendulum-v0 ../demos/model-Pendulum_a3c --num_episodes 30 --animate_not
```

### Tips for Training 

Training an RL model is typically much more difficult than training neural networks in the supervised setting. To facilliate the training there are a few regularization tricks researchers use. There are a few of these that we've incroporated:

* _Loss function regularizations_: our loss function, beside the main advantage policy-gradient term has two extra regularization terms. The first is a negative term corresponding to the entropy of policy log-probabilities and the second is a positive term corresponding to the KL distance of the new policy and the past policy. 

* _Learning rate adjustment_: We use the KL distance between the new and the old policies' probabilities assigned to actions taken in a recent episode to adjust the learning rate. The goal is to keep the KL distance betrween updated policy and the previous policy within a certain desired range.

In our experience, the desired KL parameter plays a crucial role in training and has to be often be adjusted depending on the environment. This can be done using the flag `--desired_kl`, e.g.
```
python run_AC.py --env CartPole-v0 --animate --desired_kl 0.002 
```




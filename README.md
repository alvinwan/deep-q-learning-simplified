# Deep Q-learning Neural Network (Simplified)

This repository provides a CPU, Tensorflow-less, simplified model. See the [Tensorflow implementation](http://github.com/alvinwan/deep-q-learning).

# Install

The project is written in Python 3 and is not guaranteed to successfully backport to Python 2.

(Optional) We recommend setting up a virtual environment.

```
virtualenv dqn --python=python3
source activate dqn/bin/activate
```

Say `$DQN_ROOT` is the root of your repository. Navigate to your root repository.

```
cd $DQN_ROOT
```

We need to setup our Python dependencies.

```
pip install -r requirements.txt
```

# Run

```
python run_dqn.py
```

Here are full usage instructions:

```
Usage:
    run_dqn.py [options]

Options:
    --batch-size=<size>     Batch size [default: 32]
    --envid=<envid>         Environment id [default: SpaceInvadersNoFrameskip-v3]
    --timesteps=<steps>     Number of timesteps to run [default: 40000000]
```

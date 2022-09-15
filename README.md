# Actor-Critic algorithms with Ray core

## Asynchronous Advantage Actor-Critic [^1]

In A3C multiple agents asynchronously run in parallel to generate data. This approach provides a more practical alternative to experience replay since parallelization also diversifies and decorrelates the data [^2].

there is a global network and many worker agents that each has its own parameters. Each of these agents interacts with its copy of the environment simultaneously as the other agents are interacting with their environments, and updates independently of the execution of other agents when they want to update their shared network

We use a parameter server to hold the global network, following [^3]


[^1]: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
[^2]: [Distributed Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/2011.11012.pdf)
[^3]: [Parameter server Ray documentation](https://docs.ray.io/en/latest/ray-core/examples/plot_parameter_server.html)
[^4]: [What is Gradient Accumulation in Deep Learning?](https://medium.com/towards-data-science/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)

## Todo 

- [ ] Implement the evaluation in n-step
- [ ] Implement continuous mode
- [ ] Implement the config for hyper parameter configuration
- [ ] Add explanation of a3c


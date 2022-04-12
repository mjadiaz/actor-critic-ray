# Implementing A3C with Ray Core
Progress in **distributed systems and algorithms** has been **fundamental** to the **recent success of deep learning**.

**Thus**, researchers proposed and developed **distributed** **architectures** **for DRL** empowering agents to learn **from experience faster, leverage exploration strategies, and become capable of learning a diverse set of tasks simultaneously.**

**A** novel **characteristic** **of reinforcement learning** is that the agent has an active **impact on how the data is generated** by interacting with its environment and storing experience trajectories. [^1]

## Asynchronous Advantage Actor-Critic [^2]

In A3C multiple agents asynchronously run in parallel to generate data. This approach provides a more practical alternative to experience replay since parallelization also diversifies and decorrelates the data.

there is a global network and many worker agents that each has its own parameters. Each of these agents interacts with its copy of the environment simultaneously as the other agents are interacting with their environments, and updates independently of the execution of other agents when they want to update their shared network

We use a parameter server to hold the global network, following [^3]

### The Framework


[^1]: [Distributed Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/2011.11012.pdf)
[^2]: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
[^3]: [Parameter server Ray documentation](https://docs.ray.io/en/latest/ray-core/examples/plot_parameter_server.html)
[^4]: [What is Gradient Accumulation in Deep Learning?](https://medium.com/towards-data-science/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)
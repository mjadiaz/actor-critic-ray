import ray

from networks import ActorCritic
from utils import Counter, ParameterServer, Writer
from worker import Worker, EvaluationWorker



ray.init(local_mode=False)

# Initialize Workers

updates_counter = Counter.remote()
param_server = ParameterServer.remote()
writer = Writer.remote('runs/test5_nstep')

evaluation_worker = EvaluationWorker.remote(param_server, updates_counter, writer, render=True)
workers = [Worker.remote(i, param_server, updates_counter, writer, evaluation_worker) for i in range(3)]


processes = [process.run.remote() for process in workers] 

ray.wait(processes)



ray.timeline()

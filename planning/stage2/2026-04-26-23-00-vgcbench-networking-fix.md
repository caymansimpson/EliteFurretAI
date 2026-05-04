# Context
We noticed an issue when trying to train against the `vgc_bench` baseline opponent using the Showdown WebSocket backend during reinforcement learning. 

# Before State
The external `vgcbench_runner_script` successfully connected to localhost and correctly accepted challenges, but no battles ever initialized, and the worker process timed out trying to challenge it.

# Problem
The `train.py` worker process unconditionally derived the external runner username by appending `_{server_port}` (e.g. `VGCBENCH_8000`), expecting the runner to have that exact username. However, the server launch manager only appends the port to the username if `config.num_servers > 1`. For single-server setups, the worker kept challenging `VGCBENCH_8000` while the opponent logged in simply as `VGCBENCH`.

# Solution
Ensure that `train.py` checks `if config.num_servers > 1` before appending the port to the `external_vgcbench_usernames` list. This mirrors the logic mapped out in `launch_external_vgcbench_runners`. 

# Reasoning
This fix allows `WorkerOpponentFactory` to issue valid string challenges that match the actual login username of the process started by `vgcbench_external_runner.py`, regardless of the `num_servers` configuration. 

# Next Steps / Implementation Plan
- [x] Fix the external vgcbench username derivation bug in `train.py`

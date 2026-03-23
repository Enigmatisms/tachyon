# CLI Output Preview

Using tachyon to profile [`cuda-pt` (my CUDA path tracing renderer)](https://github.com/Enigmatisms/cuda-pt), the following is generated using `tachyon profile` (end2end) mode with `--radical` setting (3 stages analysis), and the Agent API service is provided by MINIMAX-M2.5. The following are some partial screenshots.

The profiler starts with a rule-based analyzer and optimization tree:
![1.png](./1.png)

`--radical` mode has 3 stages (metrics analysis, source code and SASS level analysis, code-based optimization suggestion).
![2.png](./2.png)

Preview of some of the output of the stage 2 and 3:
![3.png](./3.png)

![4.png](./4.png)

![5.png](./5.png)

![6.png](./6.png)

`--lang zh` is available for printing in Chinese.

There will be more preview output for `tachyon chat` and `tachyon evolve` modes in the future.

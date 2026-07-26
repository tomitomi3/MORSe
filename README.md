[ English | [日本語](README_jp.md) ]

## MORSe

**M**ulti-objective **O**ptimization for **R**eservoir computing parameter **Se**arch

MORSe is a Python framework for exploring reservoir computing hyperparameters through multi-objective optimization. It uses MOEA/D as its multi-objective optimization algorithm and presents trade-offs between objectives, such as prediction or classification performance and the number of reservoir nodes, as a Pareto front. It provides parameter importance analysis, network metrics, and visualization of optimization results.

### Quick Start

#### Run Optimization

Requirements: Python 3.10 or later (Windows / Linux)

Steps:

1\. Clone the repository.

```bash
git clone https://github.com/tomitomi3/MORSe.git
cd MORSe
```

2\. Install the dependencies.

The steps for creating and activating a virtual environment are omitted.

```bash
python -m pip install -r requirements.txt
```

3\. Design the optimization target.

Optimization targets are defined as “tasks” in `src/tasks/`. See [Tasks](#tasks) for details.

4\. Run optimization with MORSe.

Run MORSe. If `realtime_visualize` in `configs/setting.json` is `false`, only optimization is performed.
The visualization server uses port 5006 by default. Use `--port` to change the port.

```bash
python main.py              # Use port 5006
python main.py --port 5007  # Specify a port
```

5\. Stop the visualization server.

To stop the server, press `Ctrl+C` in the terminal where MORSe is running.

#### Visualize Saved Results

Specify a result directory, as configured by `result_dir` in `configs/setting.json`, to visualize only the saved results.

```bash
python main.py Result
```

### Tasks

#### Sample Tasks (see the `src/tasks` directory)

Copy the configuration file you want to use to `configs/setting.json`, then run MORSe.

| Configuration file | Task | Objectives |
|---|---|---|
| `sample_narma10_prediction_activation.json` | NARMA10 prediction | Minimize the number of reservoir nodes and mean NRMSE |
| `sample_narma10_prediction_robust.json` | NARMA10 prediction (multiple seeds) | Minimize NRMSE variance and mean NRMSE |
| `sample_spoken_digit_recognition.json` | Spoken digit classification (data download required) | Minimize the number of reservoir nodes and maximize mean accuracy |
| `sample_zdt1_mv.json` | ZDT1 mixed-variable benchmark | Minimize `f1` and `f2` |
| `sample_zdt1_mv_rbring.json` | ZDT1 mixed-variable Rosenbrock Ring benchmark | Minimize `f1` and `f2` |
| `sample_zdt2_mv.json` | ZDT2 mixed-variable benchmark | Minimize `f1` and `f2` |
| `sample_zdt3_mv.json` | ZDT3 mixed-variable benchmark | Minimize `f1` and `f2` |

#### Add a Custom Task

A custom task defines decision variables (continuous, integer, or categorical), objective functions, and optimization directions. Add one as follows:

1. Create `src/tasks/<task_name>/`
2. Implement a class that inherits from `BaseTask`
3. Export the task class from `__init__.py`
4. Set the directory name in `task.name` in `configs/setting.json`

Implement the following four `BaseTask` methods:

```python
objective_function(params)  # Return objective values and optional NetworkMetrics
get_objective_names()       # Return objective names
get_directions()            # Define minimization or maximization
get_variables()             # Define decision variables
```

The optimizer handles sign conversion of objective values.

### Mixed-Variable Multi-Objective Optimization Benchmark Functions

ZDT1–ZDT3 [7], commonly used as multi-objective optimization benchmarks, are extended to problems containing continuous, integer, and categorical variables. Their known Pareto fronts enable quantitative evaluation of search performance.

- Continuous variables: $x_i \in [0, 1]\ (i = 1, \dots, N_x)$
- Integer variables: $z_i \in \lbrace 0, 1, \dots, K_z - 1 \rbrace\ (i = 1, \dots, N_z)$, normalized as $\tilde{z}_i = z_i / (K_z - 1)$
- Categorical variables: $c_i\ (i = 1, \dots, N_c)$, represented by a one-hot vector $\mathbf{c}_i = (c_{i,1}, \dots, c_{i,K_c})$ and normalized as $\tilde{c}_i = 1 - c_{i,1}$

#### ZDT1-MV / ZDT2-MV / ZDT3-MV (`MV` stands for Mixed Variable)

$f_1$ and $g$ are common to all three functions.

$$f_1 = \frac{x_1 + \tilde{z}_1 + \tilde{c}_1}{3}$$

$$g = 1 + 3 \left( \frac{1}{N_x - 1} \sum_{i=2}^{N_x} x_i + \frac{1}{N_z - 1} \sum_{i=2}^{N_z} \tilde{z}_i + \frac{1}{N_c - 1} \sum_{i=2}^{N_c} \tilde{c}_i \right)$$

$f_2$ determines the shape of the Pareto front.

ZDT1-MV:

$$f_2 = g \left( 1 - \sqrt{f_1 / g} \right)$$

ZDT2-MV:

$$f_2 = g \left( 1 - \left( f_1 / g \right)^2 \right)$$

ZDT3-MV:

$$f_2 = g \left( 1 - \sqrt{f_1 / g} - \frac{f_1}{g} \sin(10 \pi f_1) \right)$$

#### ZDT1-MV-RBRing (Extension with Variable Interactions)

To evaluate the effects of interactions between variables, the Rosenbrock function structure is incorporated into ZDT1-MV. Product terms between variables are added to $f_1$, while $g$ is composed of Rosenbrock terms connecting the remaining variables in a ring.

$$f_1 = \frac{x_1 + \tilde{z}_1 + \tilde{c}_1 + x_1 \tilde{z}_1 + \tilde{z}_1 \tilde{c}_1 + \tilde{c}_1 x_1}{6}$$

The remaining decision variables are combined into a vector $\mathbf{u}$, where $n$ is the number of elements and $u_{n+1} \equiv u_1$.

$$\mathbf{u} = \left( x_2, \dots, x_{N_x},\ \tilde{z}_2, \dots, \tilde{z}_{N_z},\ \tilde{c}_2, \dots, \tilde{c}_{N_c} \right)$$

$$g = 1 + 9 \sum_{i=1}^{n} \left\lbrace 100 \left( u_{i+1} - u_i^2 \right)^2 + \left( u_i - 1 \right)^2 \right\rbrace$$

$$f_2 = g \left( 1 - \sqrt{f_1 / g} \right)$$

### AI-Assisted Development

This project uses AI coding assistance from OpenAI Codex and Anthropic Claude Code, particularly for the visualization UI, refactoring, and testing. Developers review, correct, and test AI-generated changes.

### References

[1] 田中剛平, 中根了昌, 廣瀬明. リザバーコンピューティング: 時系列パターン認識のための高速機械学習の理論とハードウェア. 森北出版, 2021.

[2] ZHANG, Qingfu; LI, Hui. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 2007, 11.6: 712-731.

[3] 小野功, 佐藤浩, 小林重信. 単峰性正規分布交叉UNDXを用いた実数値GAによる関数最適化. 人工知能学会誌, 1999, 14.6: 1146-1155.

[4] KITA, Hajime; ONO, Isao; KOBAYASHI, Shigenobu. Multi-parental extension of the unimodal normal distribution crossover for real-coded genetic algorithms. Transactions of the Society of Instrument and Control Engineers, 2000, 36.10: 875-883.

[5] 樋口隆英, 筒井茂義, 山村雅幸. 実数値GAにおけるシンプレクス交叉の提案. 人工知能学会論文誌, 2001, 16.1: 147-155.

[6] 阪井節子, 高濱徹行. 変数間依存性を解消する変換を導入したブレンド交叉の提案. 数理解析研究所講究録, 2018, 2078: 65-72.

[7] DEB, Kalyanmoy. Multi-objective genetic algorithms: Problem difficulties and construction of test problems. Evolutionary Computation, 1999, 7.3: 205-230.

[8] ISHIBUCHI, Hisao; MASUDA, Hiroyuki; TANIGAKI, Yuki; NOJIMA, Yusuke. Modified distance calculation in generational distance and inverted generational distance. In: Evolutionary Multi-Criterion Optimization (EMO 2015), Lecture Notes in Computer Science, vol. 9019. Cham: Springer International Publishing, 2015, pp. 110-125.

[9] 森下光之助. 機械学習を解釈する技術: 予測力と説明力を両立する実践テクニック. 技術評論社, 2021.

[10] WATANABE, Shuhei; BANSAL, Archit; HUTTER, Frank. PED-ANOVA: Efficiently quantifying hyperparameter importance in arbitrary subspaces. arXiv preprint arXiv:2304.10255, 2023. See [ped-anova](https://github.com/nabenabe0928/ped-anova) for the implementation reference.

### Acknowledgments

The development of this software was supported by the Information-technology Promotion Agency, Japan (IPA), under the [2025 MITOU Target Program (Software Development Using Reservoir Computing Technologies)](https://www.ipa.go.jp/jinzai/mitou/target/2025_reservoir/index.html).

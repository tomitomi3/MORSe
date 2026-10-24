<div align="center">

[ [English](README.md) | 日本語 ]

## MORSe

**M**ulti-objective **O**ptimization for **R**eservoir computing parameter **Se**arch

</div>

MORSeは、リザバーコンピューティングのハイパーパラメータを多目的最適化で探索するPythonフレームワークである。多目的最適化アルゴリズムにMOEA/Dを用いて、予測・分類性能とリザバーノード数などのトレードオフをパレートフロントとして提示する。パラメータ重要度、ネットワーク指標、最適化結果の可視化を提供する。

### クイックスタート

#### 最適化を実行する

動作環境：Python 3.10以上（Windows / Linux）

手順：

1\. リポジトリを取得する。

```bash
git clone https://github.com/tomitomi3/MORSe.git
cd MORSe
```

2\. 依存パッケージをインストールする。

仮想環境の作成・有効化の手順は省略する。

```bash
python -m pip install -r requirements.txt
```

3\. 最適化対象を設計する。

最適化対象は「タスク」として`src/tasks/`に定義する。詳細は[タスクについて](#タスクについて)を参照する。

4\. MORSeによる最適化を実行する。

MORSeを実行する。`configs/setting.json`の`realtime_visualize`が`false`なら最適化のみを行う。
可視化サーバーはデフォルトで5006を使用する。ポートは`--port`で変更できる。

```bash
python main.py              # ポート5006
python main.py --port 5007  # ポートを指定
```

5\. 可視化サーバーを終了する。

終了するには、実行中のターミナルで`Ctrl+C`を押す。

#### 保存済み結果を可視化する

結果ディレクトリ（`configs/setting.json`の`result_dir`で指定）を指定すると、保存済み結果のみを可視化できる。

```bash
python main.py Result
```

### タスクについて

#### サンプルタスク（`src/tasks`フォルダ参照）

使用する設定ファイルを`configs/setting.json`へコピーして実行する。

| 設定ファイル | タスク | 目的関数 |
|---|---|---|
| `sample_narma10_prediction_activation.json` | NARMA10予測 | リザバーノード数の最小化、平均NRMSEの最小化 |
| `sample_narma10_prediction_robust.json` | NARMA10予測（複数seed） | NRMSE分散の最小化、平均NRMSEの最小化 |
| `sample_spoken_digit_recognition.json` | 発話数字分類（要データ取得） | リザバーノード数の最小化、平均正解率の最大化 |
| `sample_zdt1_mv.json` | ZDT1混合変数ベンチマーク | `f1`、`f2`の最小化 |
| `sample_zdt1_mv_rbring.json` | ZDT1混合変数Rosenbrock Ringベンチマーク | `f1`、`f2`の最小化 |
| `sample_zdt2_mv.json` | ZDT2混合変数ベンチマーク | `f1`、`f2`の最小化 |
| `sample_zdt3_mv.json` | ZDT3混合変数ベンチマーク | `f1`、`f2`の最小化 |

#### 独自にタスクを追加する方法

独自タスクでは、決定変数（連続・整数・カテゴリ）、目的関数、最適化方向を定義する。追加手順は次のとおり。

1. `src/tasks/<task_name>/`を作成する
2. `BaseTask`を継承したクラスを実装する
3. `__init__.py`からタスククラスを公開する
4. `configs/setting.json`の`task.name`にフォルダ名を指定する

`BaseTask`の以下の4メソッドを実装する。

```python
objective_function(params)  # 目的関数値と任意のNetworkMetricsを返す
get_objective_names()       # 目的関数名を返す
get_directions()            # 最小化／最大化を定義する
get_variables()             # 決定変数を定義する
```

目的関数値の符号反転は最適化側で行う。

### 混合変数多目的最適化ベンチマーク関数について

多目的最適化アルゴリズムのベンチマークに使われるZDT1〜ZDT3[7]を、連続・整数・カテゴリ変数が混在する問題へ拡張した。パレートフロントが既知であり、探索性能を定量評価できる。

- 連続変数 $x_i \in [0, 1]\ (i = 1, \dots, N_x)$
- 整数変数 $z_i \in \lbrace 0, 1, \dots, K_z - 1 \rbrace\ (i = 1, \dots, N_z)$ を $\tilde{z}_i = z_i / (K_z - 1)$ で正規化
- カテゴリ変数 $c_i\ (i = 1, \dots, N_c)$ をone-hotベクトル $`\mathbf{c}_i = (c_{i,1}, \dots, c_{i,K_c})`$ で表し、$`\tilde{c}_i = 1 - c_{i,1}`$ で正規化

#### ZDT1-MV / ZDT2-MV / ZDT3-MV （`MV`は Mixed Variable の略）

$f_1$ と $g$ は3つで共通である。

$$f_1 = \frac{x_1 + \tilde{z}_1 + \tilde{c}_1}{3}$$

$$g = 1 + 3 \left( \frac{1}{N_x - 1} \sum_{i=2}^{N_x} x_i + \frac{1}{N_z - 1} \sum_{i=2}^{N_z} \tilde{z}_i + \frac{1}{N_c - 1} \sum_{i=2}^{N_c} \tilde{c}_i \right)$$

$f_2$ はパレートフロントの形状を決める。

ZDT1-MV:

$$f_2 = g \left( 1 - \sqrt{f_1 / g} \right)$$

ZDT2-MV:

$$f_2 = g \left( 1 - \left( f_1 / g \right)^2 \right)$$

ZDT3-MV:

$$f_2 = g \left( 1 - \sqrt{f_1 / g} - \frac{f_1}{g} \sin(10 \pi f_1) \right)$$

#### ZDT1-MV-RBRing（変数間相互作用の拡張）

変数間相互作用の影響を評価するため、Rosenbrock関数の構造をZDT1-MVへ組み込んだ。$f_1$には変数間の積を加え、$g$は残りの変数を環状に接続したRosenbrock項で構成する。

$$f_1 = \frac{x_1 + \tilde{z}_1 + \tilde{c}_1 + x_1 \tilde{z}_1 + \tilde{z}_1 \tilde{c}_1 + \tilde{c}_1 x_1}{6}$$

残りの決定変数をベクトル $\mathbf{u}$ にまとめる（$n$は要素数、$u_{n+1} \equiv u_1$）。

$$\mathbf{u} = \left( x_2, \dots, x_{N_x},\ \tilde{z}_2, \dots, \tilde{z}_{N_z},\ \tilde{c}_2, \dots, \tilde{c}_{N_c} \right)$$

$$g = 1 + 9 \sum_{i=1}^{n} \left\lbrace 100 \left( u_{i+1} - u_i^2 \right)^2 + \left( u_i - 1 \right)^2 \right\rbrace$$

$$f_2 = g \left( 1 - \sqrt{f_1 / g} \right)$$

### AIによる開発支援

本ソフトウェアの開発にはOpenAI CodexとAnthropic Claude CodeによるAIコーディング支援を利用している。特に、可視化UI、リファクタリング、テストなどに用いている。AIによる変更は開発者が確認・修正・テストしている。

### 参考文献

[1] 田中剛平, 中根了昌, 廣瀬明. リザバーコンピューティング: 時系列パターン認識のための高速機械学習の理論とハードウェア. 森北出版, 2021.

[2] ZHANG, Qingfu; LI, Hui. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on Evolutionary Computation, 2007, 11.6: 712-731.

[3] 小野功, 佐藤浩, 小林重信. 単峰性正規分布交叉UNDXを用いた実数値GAによる関数最適化. 人工知能学会誌, 1999, 14.6: 1146-1155.

[4] KITA, Hajime; ONO, Isao; KOBAYASHI, Shigenobu. Multi-parental extension of the unimodal normal distribution crossover for real-coded genetic algorithms. Transactions of the Society of Instrument and Control Engineers, 2000, 36.10: 875-883.

[5] 樋口隆英, 筒井茂義, 山村雅幸. 実数値GAにおけるシンプレクス交叉の提案. 人工知能学会論文誌, 2001, 16.1: 147-155.

[6] 阪井節子, 高濱徹行. 変数間依存性を解消する変換を導入したブレンド交叉の提案. 数理解析研究所講究録, 2018, 2078: 65-72.

[7] DEB, Kalyanmoy. Multi-objective genetic algorithms: Problem difficulties and construction of test problems. Evolutionary Computation, 1999, 7.3: 205-230.

[8] ISHIBUCHI, Hisao; MASUDA, Hiroyuki; TANIGAKI, Yuki; NOJIMA, Yusuke. Modified distance calculation in generational distance and inverted generational distance. In: Evolutionary Multi-Criterion Optimization (EMO 2015), Lecture Notes in Computer Science, vol. 9019. Cham: Springer International Publishing, 2015, pp. 110-125.

[9] 森下光之助. 機械学習を解釈する技術: 予測力と説明力を両立する実践テクニック. 技術評論社, 2021.

[10] WATANABE, Shuhei; BANSAL, Archit; HUTTER, Frank. PED-ANOVA: Efficiently quantifying hyperparameter importance in arbitrary subspaces. arXiv preprint arXiv:2304.10255, 2023. 実装は[ped-anova](https://github.com/nabenabe0928/ped-anova)を参照

### 謝辞

本ソフトウェアの開発は、情報処理推進機構（IPA）の[2025年度 未踏ターゲット事業（リザバーコンピューティング技術を活用したソフトウェア開発分野）](https://www.ipa.go.jp/jinzai/mitou/target/2025_reservoir/index.html)の支援を受けて実施した。

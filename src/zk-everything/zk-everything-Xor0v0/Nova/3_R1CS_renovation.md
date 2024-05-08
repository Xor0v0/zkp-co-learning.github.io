# R1CS Renovation

> 作者: Xor0v0

Nova 是一个针对 R1CS Arithmetization 的递归证明系统，当然目前电路算术化 Arithmetization 方式还有很多，比如 PLONKish，AIR，CCS 等，也出现了针对这些算术化的 Nova 变体。本文只讨论 Nova 原始论文，因此本小节详细介绍 R1CS 算术化及其松弛版本。

### 1. Stantard R1CS

首先需要理解：Arithmetization 其实就是把待证明陈述转化为 [Circuit-Satisfiability](https://en.wikipedia.org/wiki/Circuit_satisfiability_problem)（C-SAT）问题。C-SAT 问题是一个经典的布尔可满足性问题，即确定在给定的布尔电路中是否存在一组输入值，使得电路的输出为真。**C-SAT是一个NP-Complete问题，目前尚未找到 PPT 算法可以解决它**。

研究人员已经证明：

- 通过增加常数级的电路门和电路深度，任何布尔电路可以转换为算数电路。
- 任意的 C-SAT 问题都可以转换为 R1CS 可满足问题。

所以，当我们证明一个电路可满足问题时，实质上可以转换为 R1CS 可满足问题。那么 R1CS 是什么？

R1CS 系统全称一阶约束系统，它实质上是把电路转化为一个**矩阵等式**。具体而言，考虑如下 `isZero`电路：

```circom
template IsZero() {
	signal input in;
	signal output out;
	
	signal inv;
	inv <-- in != 0 ? 1/in : 0;
	out <== -in * inv + 1;
	in * out === 0;
}
```

将其转换成电路有：

<div align=center><img src="./imgs/nova3.png" style="zoom:80%;" /></div>

注意在 R1CS 算术化中，电路的每一根导线都定义一个名字，记为 $x_i$ ，但是正如你所见，以常数作为输入的导线没有标记，这是因为在矩阵中有专门的一列（ $x_0$ ）供其使用，并且同一个输入的导线标记是相同的，比如 in 连接两个导线。

然后以每个乘法门为中心，根据导线的输入构造矩阵 $L, R, O$ ，分别对应乘法门的左输入、右输入和输出导线。

比如左输入矩阵：

$$
L=\begin{pmatrix}
0&1&0&0&0&0\\
0&0&1&0&0&0\\
0&1&0&0&0&0
\end{pmatrix}
$$

不难发现，矩阵的列数对应导线标记，矩阵的行数对应乘法门的个数。即 $L,R,O$ 矩阵的都是 $m\times n$ 的，$m$ 表示乘法门的个数， $n$ 表示输入输出导线的条数。

R1CS 可满足问题是说，对于给定的上述代表空白状态电路的三个矩阵，存在一个 witness 向量 $\vec{z}$（即对每根导线赋值），满足： $L\vec{z}*R\vec{z}-O\vec{z}=0$ 。其中矩阵与向量之间是点乘（省略，最基本的矩阵运算）， `*` 表示hadamard product，表示两个矩阵对应元素相乘。

对于向量 $\vec{z}$ ，它表示一组见证 witness ，它见证了整个计算过程的完成，保证 Prover 诚实地运行了整个电路（完成了整个计算过程），不然它无法得到每个导线的值。显然，**每条导线上的值就是对计算过程的见证**。特别的，规定 $x_0=1$ 恒成立。

安全性假设：由于 C-SAT 问题等价于 R1CS 可满足问题，且 C-SAT 问题是 NPC 问题，因此拥有 PPT 算力的 Prover 无法构造出满足要求的 witness 向量，除非它诚实地运行整个电路。

数学语言表述 R1CS 可满足问题：一阶约束系统 R1CS 是七元组 $(\mathbb{F},\pmb{A,B,C},\vec{io}, m,n)$ 。 其中 $\vec{io}$ 是公共输入输出向量， $\pmb{A,B,C}$ 是三个 $m\times m$ 的矩阵， $n$ 是所有矩阵中非 0 值的最大数目。称 R1CS 问题是可满足的，当且仅当对于一个 R1CS 元组，存在证据 $\vec{w}$ 使得 $(\pmb{A}\vec{z})\odot(\pmb{B}\vec{z})=\pmb{C}\vec{z}$ 其中 $\vec{z}=(\vec{io},1,\vec{w})^T$ 。其中 $\odot$ 表示hadamard product，两个矩阵元素对应相乘。

其中，定义R1CS 的 Instance 为 $\mathcal{I} = \vec{io}$ ，因为对于空白状态电路，这样一组公共输入输出唯一确定了电路的运行状态；定义 $\mathcal{W} = \vec{w}$ 为R1CS 的 Witness，表示电路的运行状态。因此 $\vec{z}=({\mathcal{I}, 1, \mathcal{W}})^T$ 。

> 这里有人可能疑惑为什么 R1CS 矩阵的定义是方阵 $m\times m$，而不是我们之前所描述的 $m\times n$ ？
> 
> 郭老师：这是由于在实际实现中，需要把矩阵按照 2 的幂进行对齐，而且大概率会对齐到同一个数 $m$ ，因此就将其简单表示为方阵。而 $n$ 表示矩阵中的非零值，这个实际意义还没有研究过。

以上就是标准的 R1CS 算术化，它被广泛运用到很多零知识证明系统中，如 Groth16，Spartan。

### 2. Relaxed R1CS

标准的 R1CS 算术化对于单个计算任务是适用的，但是当处理多个计算任务时，情况变得复杂起来。

假设对同一个电路 $F$ 计算 $n$ 次，那么意味着要对 R1CS 的矩阵等式进行 n 次证明与验证。Nova的核心思想是折叠（Folding），意思是：把 $n$ 次运行的 witness 向量 $\vec{v}_1, \dots, \vec{v}_n$ 折叠压缩成一个 $\vec{v}^*$ ，使 $\vec{v}^*$ 和 R1CS 矩阵仍然满足某个矩阵等式。这样我们就可以一次性证明 $n$ 次计算的正确性了。

Folding：对于两个数 $(a_1, a_2)$ 的折叠是很简单的，只需要随机选择一个随机值 $r$ ，则二者就可以折叠成 $a_1+ra_2$ 。这样的情况不适用于对 witness 向量的压缩：简单起见，考虑两个向量 $\vec{w}_1,\vec{w}_2$ 。我们随机选取一个随机值 $r$ ，计算压缩之后的向量为： $\vec{w}^* =\vec{w}_1+r\vec{w}_2$ 。但是问题在于，折叠之后的向量不满足之前的矩阵等式了，即 $(\pmb{A}\vec{z}^* )\odot(\pmb{B}\vec{z}^* )\neq\pmb{C}\vec{z}^*$ 。

简单验证：

$$
\begin{align*}
LHS: (\pmb{A}\vec{z}^* )\odot(\pmb{B}\vec{z}^* )&=\pmb{A}(\vec{z}_1+r\vec{z}_2)\odot \pmb{B}(\vec{z}_1+r\vec{z}_)\\
&=\pmb{A}\vec{z}_1\odot \pmb{B}\vec{z}_1 + r^2 \pmb{A}\vec{z}_2\odot \pmb{B}\vec{z}_2 + r (\pmb{A}\vec{z}_2\odot \pmb{B}\vec{z}_1 + \pmb{A}\vec{z}_1\odot \pmb{B}\vec{z}_2)\\
&=\pmb{C}\vec{z}_1 + r^2 \pmb{C}\vec{z}_2 + {\color{red}r (\pmb{A}\vec{z}_2\odot \pmb{B}\vec{z}_1 + \pmb{A}\vec{z}_1\odot \pmb{B}\vec{z}_2)}
\end{align*}
$$

红色部分就是所谓的交叉项（cross term），这显然不是我们想要的结果。

因此没办法直接对标准的 R1CS 进行 fold，Nova提出了一种松弛版本（Relaxed）的 R1CS，将矩阵等式变化为： $(\pmb{A}\vec{z} )\odot(\pmb{B}\vec{z} )=u (\pmb{C}\vec{z}) + \vec{E}$ 。其中 $u\in \mathbb{F}, \vec{E}\in \mathbb{F}^m$。引入 $\vec{E}$ 主要是为了收集红色部分的交叉项，它被称为误差向量（Error vector）。

这个 Relaxed R1CS 等式满足定理： 如果 $(\pmb{A}\vec{z}_1 )\odot(\pmb{B}\vec{z}_1 )=u_1 (\pmb{C}\vec{z}_1) + \vec{E}_1$ ， $(\pmb{A}\vec{z}_2 )\odot(\pmb{B}\vec{z}_2 )=u_2 (\pmb{C}\vec{z}_2) + \vec{E}_2$ ，那么折叠之后 witness 向量 $\vec{z}^* $ 满足 $(\pmb{A}\vec{z}^* )\odot(\pmb{B}\vec{z}^* )=u^* (\pmb{C}\vec{z}^*) + \vec{E}^* $ ，其中 $u^* = u_1 + ru_2, \vec{z}^* = \vec{z}_1 + r\vec{z}_2, \vec{E}^* =\vec{E}_1 + r^2 \vec{E}_2 + r \vec{T}, \vec{T}=\pmb{A}\vec{z}_2\odot \pmb{B}\vec{z}_1 + \pmb{A}\vec{z}_1\odot \pmb{B}\vec{z}_2 - u_1 (C\vec{z}_2) - u_2(C\vec{z}_1)$ 。（自行验证正确性）

我们定义 Relaxed R1CS Instance： $\mathcal{I}=(u, \vec{io}, \vec{E})$ ，Relaxed R1CS Witness $\mathcal{W}=\vec{w}$ ，相应地 $\vec{z}=({\mathcal{I}, 1, \mathcal{W}})^T $ 。

但是这样又引入了新问题：在计算 $\vec{z}^* = \vec{z}_1 + r\vec{z}_2$ 时，由于 $\vec{z}$ 中包含了 $\mathcal{I}$ ， $\mathcal{I}$ 中存在 $\vec{E}$ ，而 $\vec{E}$ 中又包括 $\vec{T} = \pmb{A}\vec{z}_2\odot \pmb{B}\vec{z}_1 + \pmb{A}\vec{z}_1\odot \pmb{B}\vec{z}_2 - u_1 (C\vec{z}_2) - u_2(C\vec{z}_1)$ ，注意到 $\vec{z}_i$ 中是需要 $\mathcal{W}_i$ 的。

> 为什么 Instance 里面不能引入上一次电路运行的 witness 向量呢？
>
> 如果 Instance 里面有上一次电路运行的 witness （包含在 $\vec{T}$ 中），那么折叠的证明者（folding prover）必须向验证者（folding verifier）提供见证以计算 $\vec{E}$ 。这使得折叠方案不是 non-trivial （因为 Verifier 的工作量较大，且通信量增大），也不是零知识的（Verifier 知道 witnesses ）。

### 3. Committed Relaxed R1CS

由于我们不想让 Relaxed R1CS Instance 中出现上一次电路运行的 witness 向量，因此又对 Relaxed R1CS 继续改进，改进点就是把 Instance 中的 $\vec{E}$ 变成 $\vec{E}$ 的承诺，这样会就不会引入了。于是，Relaxed R1CS Instance 变为 $\mathcal{I}=(u, \vec{io}, [\vec{E}], [\vec{w}])$ ，而 Relaxed R1CS Witness 变为 $\mathcal{W}=(E, \vec{w})$。

注意这里引入了一个 commitment scheme，意味着在 Prover 自己在 Folding 的过程中还需要设计对承诺的验证（即下一章阐述的第一对 Prover-Verifier）。


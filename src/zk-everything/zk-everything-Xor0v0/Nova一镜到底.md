# Nova一镜到底

## 0x00 动机

共学社区产出了很多的Nova相关文章，但是总感觉没串在一起，对于我这种愚笨的小白来说还是不太友好。

因此，我想从一位小白的视角描述我学习Nova的过程，当然不可避免首先要学习社区中大佬们已有的解读文章。本文主要还是把Nova的脉络理清楚，然后补充一些小白视角下的疑问解答（请教了不少老师）。从头到尾捋一遍也有助于我对Nova的理解。

希望大家各有所获！！

## 0x01 问题的起源

### ZK证明系统

关于zk证明系统的来龙去脉，由于不是本文的重点，就不详细介绍了，相信看这篇文章的同志多少还是对ZK证明系统有所认知的。

在这里主要是从**主观**的角度描述ZK证明系统在工业界的可能热门迭代方向：**无需Trusted setup**和**优化证明效率**。当然，ZKP方向当然不仅仅只有这两个方向，理论界还有很多别的发展方向，在此不做讨论。

其中，**优化证明效率**方向我将其分为两个层面，如下表：

<img src="./imgs/nova1.png" style="zoom:75%;" />

解释：

1. 协议层的优化方向分为lookup和Acc/Folding（也可以说Recursion），但是这两者不是互斥的选项，意味着可以同时采用这两种方式构建加速证明效率的方案。
2. 执行层的优化方向分为分布式并行计算和硬件加速，同样的，此二者也不是互斥的选项，可以同时采用分布式并行计算和硬件加速，参考2023年zPrice的[PLONK-DIZK](https://www.zprize.io/prizes/plonk-dizk-gpu-acceleration)。

可以看到，Nova是作为一种协议层的加速证明效率（递归证明）方案出现的。

### 【Why】为什么需要递归证明？

当使用ZK证明系统时，首先需要注意：**一个复杂且庞大的计算在单个证明系统中证明是低效且不切实际的。**这是因为我们需要把计算任务转换成电路，然后把电路、公共输入以及见证输入到证明系统中生成证明。当电路规模增加到一定数量级之后，证明速率会大大降低，甚至证明者由于实际内存限制根本无法生成证明。

再者，需要注意我们通常要求证明系统的**验证复杂度不能太高**，也就是所谓的简洁性 succinctness，在ZKP上下文中，简洁性一般指电路规模的多项式对数（poly-logarithmic）级别。【多项式对数是一种复杂度计算术语，表示为 $n^k\log^m n$】

在上述两点的条件限制下，如何对一个复杂且庞大的计算在单个证明系统中进行证明，并且保证验证者的验证成本在可接受范围内呢？

研究人员提出了三种「高效」证明思想：

1. 证明组合（Proof Composition）：将使用不同特点的证明系统组合起来以生成最终证明。比如可以先使用一个快速的证明系统，但是可能生成的proof size较大。为了减小proof size，可以再组合一个较慢的证明系统，用于证明「快证明系统」的证明可以通过验证者验证。注意，第二个「慢证明系统」其实证明的是第一个「快证明系统」的验证者电路，而验证者电路一般复杂度是较低的。因此，通过证明组合，我们可以实现用相对较快的方式证明一个复杂的计算电路的运行过程。

2. 证明聚合（Proof Aggregation）：将多个proof聚合成一个单一的proof。比如zkEVM包含了许多不同的电路，如EVM电路、RAM电路、存储电路等。在实践中，验证所有这些证明在区块链上是非常昂贵的。因此，我们使用证明聚合，即使用一个证明来证明这些多个子电路证明是有效的。这个单一的证明在区块链上更容易被验证。因此，这种证明思想下，需要首先各自运行证明系统分别生成证明，最后使用一个聚合证明的证明系统以证明各个证明的验证有效性和聚合的合法性。这种时间复杂度还是很高的。

3. 证明递归（Proof Recursion）：这是一种更为高明的证明思想，它旨在把复杂业务电路分割成相对小的计算电路，然后构建一连串验证者电路（**也就是说每次在进行业务电路运算的时候，顺带着把上一次的电路运行证明进行部分或者全部验证**），这样可以有效的**摊销**验证成本。如果每次运行的业务电路是一样的（即IVC，见后文解释），那么这样每一步我们可以生成一个比前一个证明更容易验证的证明。最终，我们只需要用最低的复杂度来验证最终的证明。

   注意上述加粗语句中，部分验证对应于Accumulator，全部验证对应于Folding。

通过上面的解释可知：递归证明可以有效的证明一个复杂且庞大的电路。

### 【What】Nova可以应用于什么领域？

在说回Nova本身，它是一种基于折叠Folding的证明技术，它的最主要用途就是实现高效的IVC。

而Nova的变种方案，则是在Nova基础上更进一步了：

SuperNova：它可以实现每一步证明不同的业务电路。

【TODO】: HyperNova, ProtoStar, UltraPlonkova.....

## 0x02 Cyrpto Primitives

本节主要介绍 Nova 证明系统中需要使用到的密码学原语。首先是两个算术电路友好型哈希函数，pedersen 和 poseidon ，

### Pederson

哈希函数可以将任意输入，映射为固定长度的输出。Pederson 哈希函数其实是非常自然的一种构造，它把消息的比特串按照固定长度分组编码，然后根据随机选择的若干群生成器，对应进行群元素的点乘，相加即可得到最终的哈希值。Pedersen 哈希有以下特点：

- 编码函数 `encoding()`：用于对消息分组编码，实践中一般输出一个域元素。
- 随机性：需要对群生成元进行随机取样，才能确保安全性。
- 输出：最终输出的哈希摘要实质上是一个群元素。

更正式地表达：给定消息是规模为 $k * r$ 的比特串，即可以按照 $r$ 长度将其分为 $k$ 组。然后随机在指定配对友好型椭圆曲线的素数阶子群中选择 $k$ 个生成元 $g_1, g_2, \dots, g_k$ 。这些生成元必须是随机均匀取样，使得两两之间的关系是无从知晓的。最后计算 pedersen hash $h=m_1g_1+m_2g_2+\dots+m_kg_k$ 。其中如何把 $r$ 大小的 bits 分组编码成 $m_i$ 需要设计成一个 encoding_function （可以简单地将其转化成 ASCII 码，也可以像zcash一样自定义）。

下面是 sage 的 PoC 代码：

```sage
# Definition
F = GF(127)
E = EllipticCurve(F, [F(1), F(42)])    

assert E.order().is_prime()
print(E.order())

G1 = E(1, 60)
G2 = E(2, 59)
GENERATORS = [G1, G2]

def IdentityEncode(lebs):
    return int(lebs[::-1], 2)

def EncodeMessage(msg, encode_function):
    return [encode_function(x) for x in msg]

def GenericPedersenHash(generators, encoded_chunks):
    assert len(generators) == len(encoded_chunks), "Incompatible lengths"
    
    res = E(0)
    
    for chunk, generator in zip(encoded_chunks, generators):
        res += chunk * generator
        
    return res

def PedersenHash(msg, encode_function=IdentityEncode, generators=GENERATORS):
    encoded_msg = EncodeMessage(msg, encode_function)
    return GenericPedersenHash(generators, encoded_msg)

def ZcashEncode(bin_value):
    r"""
    Zcash's encoding function (·): encodes `bin_value`, a binary value in little-endian bit order to an element
    in the range {-(r-1)/2 .. (r-1)/2}\{0} with r the subgroup order
    """
    
    def enc(b):
        """
        Zcash's 3-bit signed encoding function
        """
        return (1 - 2 * int(b[2])) * (1 + int(b[0]) + 2 * int(b[1]))
    
    assert len(bin_value) % 3 == 0
    
    res = 0
    for j, a in enumerate(range(0, len(bin_value), 3)):
        bchunk = bin_value[a: a+3]
        res += enc(bchunk) * (2 ** (4 * j))
    return res

def ZcashPedersenHash(message, generators=GENERATORS):
    return PedersenHash(message, encode_function=ZcashEncode, generators=generators)
        

message = ["010101", "000111"]
H = PedersenHash(message)
print(f"Hash of {message} is {H}")

assert H == 42 * G1 + 56 * G2, "Nope"
```

#### Security Requirements

1. Collision Resistence：为了实现抗碰撞这一特性，要求输入长度是固定的且编码函数必须是单射（injective）的。

   比如上述代码示例中，如果编码函数输出的标量值大于 subgroup 的阶 139，那么攻击者就有可能制造碰撞。

   ```sage
   # Collision-resistant: One should ensure its encoding is smaller than the subgroup order
   H2 = (42 + 139) * G1 + (56 + 139) * G2
   assert H == H2, "Nope"
   print(int("10110101", 2))
   ```

   如果输入可变，则也会导致碰撞，比如：

   ```sage
   message = ["010101", "111000"]
   colliding_message = [message[0] + "000", message[1][:3]]
   print(f"Colliding Message: {colliding_message}")
   # Colliding Message:  ['010101000', '111']

   assert PedersenHash(colliding_message) == PedersenHash(message)
   ```

2. Randomness：之前已经说过，必须随机挑选群生成元，确保二者之间联系是无人知晓的（即使是协议设计者）。

   如果生成元之间的关系泄漏，那么就可能造成攻击。比如：

   ```sage
   # We know the discrete logarithm of G2 with respect to G1
   assert 35 * G1 == G2
   message = ["010101", "000111"]  # M = [42, 56]
   encoded_message = EncodeMessage(message, ZcashEncode)
   print(f"Encoded Message: {encoded_message}")

   H = ZcashPedersenHash(message)
   assert 129 * G1 == H

   # We need to find a message M' = M1' || M2' such that
   # <M1'> + 35*<M2'> =  (-29 -35*63 ) = 129 mod 139
   H2 = GenericPedersenHash(GENERATORS, [17, 31])
   assert H == H2
   print(ZcashEncode("000000"))

   H3 = ZcashPedersenHash(["000000", "001100"])
   assert H == H3
   # Encoded Message:  [-29, -63]
   ```

关于 Pedersen hash 在实际应用还有很多有趣的安全问题，如果想继续学习这方面的知识，可以尝试 ZKHACK [Let's hash it out](https://zkhack.dev/events/puzzle1.html)。我也分享了带有代码的 [wp](https://github.com/Xor0v0/ZKHack-Solutions/tree/main/Let's%20Hash%20it%20out)。

### Poseidon

[Poseidon hash](https://eprint.iacr.org/2019/458.pdf) 也是一个 SNARK 友好型哈希函数，与 keccak hash 和SHA-3 hash一样，它的构造也参考了 [sponge functions](https://pdfs.semanticscholar.org/a949/02166ba971cbb6d0e31bbf4c51b000fbeae5.pdf?_ga=2.13918709.372950161.1571046613-1498580965.1561015485) 的模型。

简单来说，sponge函数（也称海绵函数）分为两个步骤，如同海绵一样，先吸收（absorbing），再挤压（squeezing）。笼统来说，其实就是先把消息分组，然后使用一个内部置换函数逐步的把这些分组消息「吸收」进来，然后再「挤压」出规定数量的哈希值。

除了那个内部置换函数，Poseidon hash还需要定义两个参数，即比率 $r$ 和容量 $c$ 。其中 $r$ 决定了吞吐量， $c$ 与安全等级有关。这意味着，当确定了固定输入的内部置换函数，实现者需要在吞吐量和安全等级之间作出取舍平衡。

<img src="./imgs/nova2.png" style="zoom:75%;" />

这里不再深究内部置换函数的细节，如有需要可自行阅读论文。我比较关心其安全问题，推荐两个深入学习材料： [审计poseidon的安全参数](https://research.nccgroup.com/2022/09/12/public-report-penumbra-labs-decaf377-implementation-and-poseidon-parameter-selection-review/)和[Scalebit CTF Roundabout](https://github.com/scalebit/zkCTF-day1/tree/main/Roundabout)。

## 0x03 R1CS改造计划

### 1. Stantard R1CS

### 2. Relaxed R1CS

### 3. Committed Relaxed R1CS

## 0x04 Basic Folding Scheme(NIFS)

这里介绍第一对Prover-Verifier

## 0x05 IVC Folding Scheme

这里介绍第二对Prover-Verifier

## 0x06 Spartan



## 0x07 Full Nova

这里介绍第三对Prover-Verifier

### Summary

总结一下Nova证明系统的特性：

- No Trusted Setup.
- No Polynomial Multiplication(No FFT!!).
- The Nova prover run time is group exp(MSM) of size $O(|F|)$, which theoretically 5-50 times faster than Groth16, Plonk, Halo, and other solutions.
- The verifier circuit of Nova is constant: 2 scalar multiplication. So the **recuision threshold** can be ....
- The proof size is $O(|\log F|)$.

## 0x08 Code Structure&Example



## 0x09 Nova+ZKML

准备把毕设搞过来简单说一下

## Reference

Vedios:

- [by 作者Srinath Setty [EN\]](https://www.youtube.com/watch?v=mY-LWXKsBLc)

- [by 郭宇@安比: A short introduction to Nova [EN\]](https://www.youtube.com/watch?v=hq-1bLVz59w&t=324s)

- [by 郭宇@安比: Nova - Recursive SNARKs without trusted setup [中文\]](https://www.youtube.com/watch?v=l19roUItyUE)

- [Nova Crash Course with **Justin Drake** [EN\]](https://www.youtube.com/watch?v=SwonTtOQzAk&t=2815s)

Papers:

- [Nova: Recursive Zero-Knowledge Arguments from Folding Schemes](https://eprint.iacr.org/2021/370.pdf)
- [Revisiting the Nova Proof System on a Cycle of Curves](https://eprint.iacr.org/2023/969)

Blogs:

- [difference between base field and scalar field](https://crypto.stackexchange.com/questions/66436/for-an-elliptic-curve-what-is-the-difference-between-the-base-field-modulus-q)
- [order of ecc](https://medium.com/asecuritysite-when-bob-met-alice/whats-the-order-in-ecc-ac8a8d5439e8)
- [Scalable Zero Knowledge via Cycles of Elliptic Curves](https://eprint.iacr.org/2014/595.pdf)
- [The halo2 book](https://zcash.github.io/halo2/index.html)
- [Nova based zk VM](https://hackmd.io/@monyverse/H1XSVmHNh#Curve-Cycling)
- https://mp.weixin.qq.com/s/WPh8b6otBdan3JHejF9zVg
- https://snowolf0620.xyz/index.php/zkp/1267.html
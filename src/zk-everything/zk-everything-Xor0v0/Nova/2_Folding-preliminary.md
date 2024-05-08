# Folding Preliminary

> 作者: Xor0v0

从本章开始我们开始沿着 `Folding` 这条脉络开始梳理. 

如第一章所述, `Folding` 操作主要针对的 Nova 的折叠方案, 这也是整个 Nova 方案最核心的创新点, 我们特意将其与后续的证明方案分开介绍. 

在介绍 Folding 协议之前, 首先必不可少地需要介绍一些密码学原语. 值得强调的是, Folding 方案本身不做电路的证明, 而是把若干相同且串联电路「折叠」成一个电路, 以便后续证明系统进行证明. 因此 Folding 其实是处理电路的过程, 其核心在于理解算术化过程.


## Crypto Primitives

### Circuit-friendly Hash

既然要研究电路, 那就不得不说电路编程中所需要的电路友好型哈希函数: Pedersen, Poseidon 和 MiMC sponge. 这里主要介绍前两种, 因为 Nova 需要.

### Pedersen Hash

哈希函数可以将任意输入，映射为固定长度的输出。Pedersen 哈希函数其实是非常自然的一种构造，它把消息的比特串按照固定长度分组编码，然后根据随机选择的若干群生成器，对应进行群元素的点乘，相加即可得到最终的哈希值。Pedersen 哈希有以下特点：

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

<div align=center><img src="./imgs/nova2.png" style="zoom:65%;" /></div>

这里不再深究内部置换函数的细节，如有需要可自行阅读论文。我比较关心其安全问题，推荐两个深入学习材料： [审计poseidon的安全参数](https://research.nccgroup.com/2022/09/12/public-report-penumbra-labs-decaf377-implementation-and-poseidon-parameter-selection-review/)和[Scalebit CTF Roundabout](https://github.com/scalebit/zkCTF-day1/tree/main/Roundabout)。


# algebrain

This is an experiment to see how well a recurrent neural network can learn to manipulate algebraic expressions.

# Architecture

The current architecture is attention-based, and closely resembles [networks used for machine translation](https://arxiv.org/abs/1409.0473). Really, this is just a simple demonstration of the power of neural attention.

# Results

The way the network works, you provide a query and it returns a result. I trained a network on the shift and scale challenges, meaning that it can accept queries of the form "shift x by ... in ..." and "scale x by ... in ...". It achieves a near perfect success rate (not perfect because some outliers are produced by the way random numbers are sampled from a Gaussian):

```
Query> shift x by 3 in (x^2-3)^3
((x-3)^2-3)^3
Query> scale x by 2 in 2*x
2*(x*2)
Query> scale x by 3 in -x
-(x*3)
Query> scale x by 4 in (x/3-2)*(x^2)
((x*4)/3-2)*((x*4)^2)
Query> shift x by 4 in ((x*4)/3-2)*((x*4)^2)
(((x-4)*4)/3-2)*(((x-4)*4)^2)
```

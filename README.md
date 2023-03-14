# torch-sum-axis

Because I already forgot https://youtu.be/PaCmpygFfXo?t=5025, explain why:
```py
>>> torch.tensor(((2,3),(4,5)))
tensor([[2, 3],
        [4, 5]])
>>> torch.tensor(((2,3),(4,5))).sum(0)
tensor([6, 8])
>>> torch.tensor(((2,3),(4,5))).sum(1)
tensor([5, 9])
```
Or:
```py
>>> np.array([[2,3], [4,5]])
array([[2, 3],
       [4, 5]])
>>> np.array([[2,3], [4,5]]).sum(0)
array([6, 8])
>>> np.array([[2,3], [4,5]]).sum(1)
array([5, 9])
```

https://stackoverflow.com/questions/40857930/how-does-numpy-sum-with-axis-work

```py
a = np.arange(30).reshape(2, 3, 5)
```
```py
>>> a
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]],

       [[15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]])
```
In particular look at `dim 0` and notice how `[[15 16 17 18 19]...` is also `dim 0`:
```
            p  p  p  p  p
            o  o  o  o  o
            s  s  s  s  s

     dim 2  0  1  2  3  4

            |  |  |  |  |
  dim 0     ↓  ↓  ↓  ↓  ↓
  ----> [[[ 0  1  2  3  4]   <---- dim 1, pos 0
  pos 0   [ 5  6  7  8  9]   <---- dim 1, pos 1
          [10 11 12 13 14]]  <---- dim 1, pos 2
  dim 0
  ---->  [[15 16 17 18 19]   <---- dim 1, pos 0
  pos 1   [20 21 22 23 24]   <---- dim 1, pos 1
          [25 26 27 28 29]]] <---- dim 1, pos 2
            ↑  ↑  ↑  ↑  ↑
            |  |  |  |  |

     dim 2  p  p  p  p  p
            o  o  o  o  o
            s  s  s  s  s

            0  1  2  3  4
```
In particular pay attention to how you think about the difference between the index in a dimension, and the dimension itself. `a[:, :, 3] # dim 2, pos 3`
```py
>>> a
array([[[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]],

       [[15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29]]])
>>> a.sum(0)
array([[15, 17, 19, 21, 23],
       [25, 27, 29, 31, 33],
       [35, 37, 39, 41, 43]])
```
Same as:
```
a[0, :, :] + \
a[1, :, :]
```
Okay you should see it now. A harder one is
```py
>>> t
tensor([[[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14]],

        [[15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29]]])
>>> t.sum(1)
tensor([[15, 18, 21, 24, 27],
        [60, 63, 66, 69, 72]])
```

```py
>>> for i in range(t.shape[1]):
...     print(t[:, i, :])
...
tensor([[ 0,  1,  2,  3,  4],
        [15, 16, 17, 18, 19]])
tensor([[ 5,  6,  7,  8,  9],
        [20, 21, 22, 23, 24]])
tensor([[10, 11, 12, 13, 14],
        [25, 26, 27, 28, 29]])
>>> t.sum(1)
tensor([[15, 18, 21, 24, 27],
        [60, 63, 66, 69, 72]])
```
The visual way I was just thinking about is is this:

```py
>>> t
tensor([[[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14]],

        [[15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29]]])
```
Strip the outtermost brackets, the outermost brackets are `dim 0`, you're doing `:` so you're going over both of them at each step. The next layer is the `dim 1`, that's the one you're indexing with `i`, that's the axis defied in `t.sum(1)`, so you take the 0th, 1st, 2nd, or the `len(t's axis 1)`. The next one is `:` too, so you take every element of the rows `[ 0,  1,  2,  3,  4]` and `[15, 16, 17, 18, 19]`.

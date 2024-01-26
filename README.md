# micrograd in C++
A header only modern C++ implementation of Autograd with inspiration taken from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).  Backpropagation (reverse-mode autodiff) is implemented over a dynamically built DAG.  A small neural network library is also implemented and tested.  The network is constructed from scalar values through the use of overloaded operator expressions (`+, -, /, *, ^`).

## Requirements
- CMake 3.22.1
- GCC 13.1.0

## Example
See the more complete test code in the `tests` directory.  In this code we create the classic single neuron given by the expression $\tanh(w \cdot x + b)$.  Calling `o.backward()` runs backpropagation on the expression, `o`, filling in gradient values.
```cpp
  // inputs x1, x2
  auto x1 = std::make_shared<Value<double>>(2.0);
  x1->setLabel("x1");
  auto x2 = std::make_shared<Value<double>>(0.0);
  x2->setLabel("x2");

  // weights w1,w2
  auto w1 = std::make_shared<Value<double>>(-3.0);
  w1->setLabel("w1");
  auto w2 = std::make_shared<Value<double>>(1.0);
  w2->setLabel("w2");

  // bias of the neuron
  auto b = std::make_shared<Value<double>>(6.8813735870195432);
  b->setLabel("b");

  auto x1w1 = x1 * w1;
  x1w1->setLabel("x1w1");
  auto x2w2 = x2 * w2;
  x2w2->setLabel("x2w2");
  auto x1w1x2w2 = x1w1 + x2w2;
  x1w1x2w2->setLabel("x1w1 + x2w2");

  auto n = x1w1x2w2 + b;
  n->setLabel("n");

  auto o = tanh(n);
  o->setLabel("o");
  o->backward();
```

## Additional Notes
The `operator^` (bitwise XOR) is overloaded to represent exponentiation similar to `std::pow`.  Unfortunately, operator precedence can not be controlled in C++ which means care must be taken with parenthesis when constructing expressions with `^` that involve other mathematical operators of higher precedence (e.g. `+, -, /, *`);

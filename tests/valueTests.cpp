#include <engine/value.h>
#include <gtest/gtest.h>

TEST(valueTests, expression1) {
  auto a = std::make_shared<Value<double>>(3.0);
  a->setLabel("a");
  auto b = a + a;
  b->setLabel("b");
  b->backward();
  EXPECT_DOUBLE_EQ(a->getGrad(), 2.0);
  EXPECT_DOUBLE_EQ(b->getGrad(), 1.0);
}

TEST(valueTests, expression2) {
  auto a = std::make_shared<Value<double>>(-2.0);
  a->setLabel("a");
  auto b = std::make_shared<Value<double>>(3.0);
  b->setLabel("b");
  auto d = a * b;
  d->setLabel("d");
  auto e = a + b;
  e->setLabel("e");
  auto f = d * e;
  f->setLabel("f");
  f->backward();
  EXPECT_DOUBLE_EQ(a->getGrad(), -3.0);
  EXPECT_DOUBLE_EQ(b->getGrad(), -8.0);
  EXPECT_DOUBLE_EQ(e->getGrad(), -6.0);
  EXPECT_DOUBLE_EQ(d->getGrad(), 1.0);
  EXPECT_DOUBLE_EQ(f->getGrad(), 1.0);
}

TEST(valueTests, expression3) {
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

  EXPECT_DOUBLE_EQ(x1->getData(), 2.0);
  EXPECT_DOUBLE_EQ(x1->getGrad(), -1.5);

  EXPECT_DOUBLE_EQ(w1->getData(), -3.0);
  EXPECT_DOUBLE_EQ(w1->getGrad(), 1.0);

  EXPECT_DOUBLE_EQ(x2->getData(), 0.0);
  EXPECT_DOUBLE_EQ(x2->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(w2->getData(), 1.0);
  EXPECT_DOUBLE_EQ(w2->getGrad(), 0.0);

  EXPECT_DOUBLE_EQ(x1w1->getData(), -6.0);
  EXPECT_DOUBLE_EQ(x1w1->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(x2w2->getData(), 0.0);
  EXPECT_DOUBLE_EQ(x2w2->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(x1w1x2w2->getData(), -6.0);
  EXPECT_DOUBLE_EQ(x1w1x2w2->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(b->getData(), 6.8813735870195432);
  EXPECT_DOUBLE_EQ(b->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(n->getData(), 0.88137358701954316);
  EXPECT_DOUBLE_EQ(n->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(o->getData(), 0.7071067811865476);
  EXPECT_DOUBLE_EQ(o->getGrad(), 1.0);
}

TEST(valueTests, expression4) {
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

  auto e = exp(2.0 * n);
  auto o = (e - 1.0) / (e + 1.0);
  o->setLabel("o");
  o->backward();

  EXPECT_DOUBLE_EQ(x1->getData(), 2.0);
  EXPECT_DOUBLE_EQ(x1->getGrad(), -1.5);

  EXPECT_DOUBLE_EQ(w1->getData(), -3.0);
  EXPECT_DOUBLE_EQ(w1->getGrad(), 1.0);

  EXPECT_DOUBLE_EQ(x2->getData(), 0.0);
  EXPECT_DOUBLE_EQ(x2->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(w2->getData(), 1.0);
  EXPECT_DOUBLE_EQ(w2->getGrad(), 0.0);

  EXPECT_DOUBLE_EQ(x1w1->getData(), -6.0);
  EXPECT_DOUBLE_EQ(x1w1->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(x2w2->getData(), 0.0);
  EXPECT_DOUBLE_EQ(x2w2->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(x1w1x2w2->getData(), -6.0);
  EXPECT_DOUBLE_EQ(x1w1x2w2->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(b->getData(), 6.8813735870195432);
  EXPECT_DOUBLE_EQ(b->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(n->getData(), 0.88137358701954316);
  EXPECT_DOUBLE_EQ(n->getGrad(), 0.5);

  EXPECT_DOUBLE_EQ(o->getData(), 0.7071067811865476);
  EXPECT_DOUBLE_EQ(o->getGrad(), 1.0);
}
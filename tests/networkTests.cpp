#include <gtest/gtest.h>
#include <network/mlp.h>

TEST(networkTests, mlp1) {
  using ValueType = Value<double>;
  std::vector<std::shared_ptr<ValueType>> x0;
  for (auto val : {2.0, 3.0, -1.0})
    x0.push_back(std::make_shared<ValueType>(val));
  std::vector<std::shared_ptr<ValueType>> x1;
  for (auto val : {3.0, -1.0, 0.5})
    x1.push_back(std::make_shared<ValueType>(val));

  auto n = MLP<double>(3, std::vector<std::size_t>{4, 4, 1});
  std::cout << n(x0).front() << std::endl;
  std::cout << n(x1).front() << std::endl;
}

TEST(networkTests, mlp2) {
  using ValueType = Value<double>;
  auto genSample = [](std::vector<double> const &samples) {
    std::vector<std::shared_ptr<ValueType>> ret;
    ret.reserve(samples.size());
    for (auto sample : samples)
      ret.push_back(std::make_shared<ValueType>(sample));
    return ret;
  };
  std::vector<std::vector<std::shared_ptr<ValueType>>> xs;
  xs.push_back(genSample({2.0, 3.0, -1.0}));
  xs.push_back(genSample({3.0, -1.0, 0.5}));
  xs.push_back(genSample({0.5, 1.0, 1.0}));
  xs.push_back(genSample({1.0, 1.0, -1.0}));

  std::vector<double> ys{1.0, -1.0, -1.0, 1.0};

  auto n = MLP<double>(3, std::vector<std::size_t>{4, 4, 1});

  // forward pass
  for (auto k = 0; k < 100; ++k) {
    std::vector<std::vector<std::shared_ptr<ValueType>>> ypreds;
    for (auto const &x : xs) {
      ypreds.push_back(n(x));
    }
    for (auto const &ypred : ypreds) {
      std::cout << std::format("ypred={}", ypred.front()) << std::endl;
    }
    auto loss = std::make_shared<ValueType>(0.0);
    double checkLoss = 0.0;
    for (auto i = 0; i < ypreds.size(); ++i) {
      checkLoss += std::pow((ypreds[i][0]->getData() - ys[i]), 2);
      loss = loss + ((ypreds[i][0] - ys[i]) ^ 2.0);
    }
    std::cout << std::format("Loss = {}", loss) << std::endl;
    std::cout << std::format("CheckLoss = {}", checkLoss) << std::endl;

    // backward pass
    n.zeroGrad();
    loss->backward();

    // SGD updated
    for (auto const &p : n.parameters())
      p->setData(p->getData() - 0.1 * p->getGrad());
    std::cout << std::format("k={}, loss={}", k, loss) << std::endl;
  }
}
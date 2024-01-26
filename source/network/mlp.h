#pragma once

#include <engine/value.h>
#include <random>
#include <vector>

template <typename T> class Module {
public:
  Module() = default;
  virtual std::vector<std::shared_ptr<Value<T>>> parameters() const = 0;
  void zeroGrad() {
    for (auto p : parameters()) {
      p->setGrad(0.0);
    }
  }
};

template <typename T> class Neuron : public Module<T> {
public:
  Neuron(std::size_t fanIn) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-1.0, 1.0);
    _w.reserve(fanIn);
    for (auto i = 0; i < fanIn; ++i) {
      _w.push_back(std::make_shared<Value<T>>(dis(gen)));
    }
    _b = std::make_shared<Value<T>>(dis(gen));
    std::cout << "Neuron: " << std::endl;
    for (auto const &w : _w)
      std::cout << std::format("\tweight={}", w->getData()) << std::endl;
    std::cout << std::format("\tbias={}", _b->getData()) << std::endl;
  }

  std::vector<std::shared_ptr<Value<T>>> parameters() const override {
    auto ret = _w;
    ret.push_back(_b);
    return ret;
  }

  std::shared_ptr<Value<T>>
  operator()(std::vector<std::shared_ptr<Value<T>>> const &x) const {
    assert(x.size() == _w.size());
    auto sum = std::make_shared<Value<T>>(_b->getData());
    for (auto i = 0; i < x.size(); ++i) {
      sum = sum + x[i] * _w[i];
    }
    return tanh(sum);
  }

private:
  std::vector<std::shared_ptr<Value<T>>> _w;
  std::shared_ptr<Value<T>> _b;
};

template <typename T> class Layer : public Module<T> {
public:
  Layer(std::size_t fanIn, std::size_t fanOut) {
    std::cout << "Building layer: " << std::endl;
    for (auto i = 0; i < fanOut; ++i) {
      _neurons.emplace_back(fanIn);
    }
  }

  std::vector<std::shared_ptr<Value<T>>> parameters() const override {
    std::vector<std::shared_ptr<Value<T>>> ret;
    for (auto const &neuron : _neurons) {
      auto params = neuron.parameters();
      ret.insert(ret.end(), params.begin(), params.end());
    }
    return ret;
  }

  std::vector<std::shared_ptr<Value<T>>>
  operator()(std::vector<std::shared_ptr<Value<T>>> const &x) {
    std::vector<std::shared_ptr<Value<T>>> ret;
    for (auto const &neuron : _neurons) {
      ret.push_back(neuron(x));
    }
    return ret;
  }

private:
  std::vector<Neuron<T>> _neurons;
};

template <typename T> class MLP : public Module<T> {
public:
  MLP(std::size_t fanIns, std::vector<std::size_t> fanOuts) {
    _layers.emplace_back(fanIns, fanOuts[0]);

    for (auto i = 0; i < fanOuts.size() - 1; ++i)
      _layers.emplace_back(fanOuts[i], fanOuts[i + 1]);
  }

  std::vector<std::shared_ptr<Value<T>>> parameters() const override {
    std::vector<std::shared_ptr<Value<T>>> ret;
    for (auto const &layer : _layers) {
      auto params = layer.parameters();
      ret.insert(ret.end(), params.begin(), params.end());
    }
    return ret;
  }

  std::vector<std::shared_ptr<Value<T>>>
  operator()(std::vector<std::shared_ptr<Value<T>>> const &x) {
    auto in = x;
    for (auto &layer : _layers) {
      in = layer(in);
    }
    return in;
  }

private:
  std::vector<Layer<T>> _layers;
};
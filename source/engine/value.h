#pragma once
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

template <typename T>
class Value : public std::enable_shared_from_this<Value<T>> {
public:
  using ChildType = std::shared_ptr<Value<T>>;
  using ChildrenType = std::vector<ChildType>;

  Value(T const &data, std::vector<ChildType> const &children = {},
        std::string const &op = "", std::string const &label = "")
      : _data(data), _children(children), _grad(0.), _op(op), _label(label) {}

  // getters and setters
  T const &getData() const { return _data; }
  void setData(T const &data) { _data = data; }

  ChildrenType const &getChildren() const { return _children; }
  void addChild(ChildType child) { _children.push_back(child); }

  T const &getGrad() const { return _grad; }
  T &getGrad() { return _grad; }
  void setGrad(T const &grad) { _grad = grad; }

  std::string const &getOp() const { return _op; }
  void setOp(char op) { _op = op; }

  std::string const &getLabel() const { return _label; }
  void setLabel(std::string const &label) { _label = label; }

  auto const &getBackward() const { return _backward; }
  template <typename F> void setBackward(F &&f) {
    _backward = std::forward<F>(f);
  }

  void backward() {
    std::vector<ChildType> nodes;
    std::unordered_set<ChildType> visited;
    dfs(this->shared_from_this(), nodes, visited);
    std::reverse(nodes.begin(), nodes.end());
    this->setGrad(1.0);
    for (auto node : nodes) {
      node->getBackward()();
    }
  }

private:
  void dfs(ChildType const &node, std::vector<ChildType> &nodes,
           std::unordered_set<ChildType> &visited) {
    visited.insert(node);
    for (auto const &child : node->getChildren()) {
      if (!visited.count(child))
        dfs(child, nodes, visited);
    }
    nodes.push_back(node);
  }

  T _data;
  ChildrenType _children;
  T _grad = 0;
  std::string _op;
  std::string _label;
  std::function<void()> _backward = []() {};
};

template <typename T>
std::shared_ptr<Value<T>> operator+(std::shared_ptr<Value<T>> lhs,
                                    std::shared_ptr<Value<T>> rhs) {
  auto ret = std::make_shared<Value<T>>(
      lhs->getData() + rhs->getData(),
      typename Value<T>::ChildrenType{lhs, rhs}, "+");
  ret->setBackward([=]() {
    lhs->getGrad() += ret->getGrad();
    rhs->getGrad() += ret->getGrad();
  });
  return ret;
}

template <typename T>
std::shared_ptr<Value<T>> operator+(typename std::shared_ptr<Value<T>> lhs,
                                    T const &rhs) {
  return lhs + std::make_shared<Value<T>>(rhs);
}

template <typename T>
std::shared_ptr<Value<T>> operator+(T const &lhs,
                                    typename std::shared_ptr<Value<T>> rhs) {
  return std::make_shared<Value<T>>(lhs) + rhs;
}

template <typename T>
std::shared_ptr<Value<T>> operator*(std::shared_ptr<Value<T>> lhs,
                                    std::shared_ptr<Value<T>> rhs) {
  auto ret = std::make_shared<Value<T>>(
      lhs->getData() * rhs->getData(),
      typename Value<T>::ChildrenType{lhs, rhs}, "*");
  ret->setBackward([=]() {
    lhs->getGrad() += rhs->getData() * ret->getGrad();
    rhs->getGrad() += lhs->getData() * ret->getGrad();
  });
  return ret;
}

template <typename T>
std::shared_ptr<Value<T>> operator*(std::shared_ptr<Value<T>> lhs,
                                    T const &rhs) {
  return lhs * std::make_shared<Value<T>>(rhs);
}

template <typename T>
std::shared_ptr<Value<T>> operator*(T const &lhs,
                                    typename std::shared_ptr<Value<T>> rhs) {
  return std::make_shared<Value<T>>(lhs) * rhs;
}

template <typename T>
std::shared_ptr<Value<T>> operator-(typename std::shared_ptr<Value<T>> lhs,
                                    typename std::shared_ptr<Value<T>> rhs) {
  return lhs + (-1.0 * rhs);
}

template <typename T>
std::shared_ptr<Value<T>> operator-(typename std::shared_ptr<Value<T>> lhs,
                                    T const &rhs) {
  return lhs + (-1.0 * std::make_shared<Value<T>>(rhs));
}

template <typename T>
std::shared_ptr<Value<T>> operator-(T const &lhs,
                                    typename std::shared_ptr<Value<T>> rhs) {
  return std::make_shared<Value<T>>(lhs) + (-1.0 * rhs);
}

template <typename T>
std::shared_ptr<Value<T>> operator^(typename std::shared_ptr<Value<T>> value,
                                    T const &power) {
  auto ret = std::make_shared<Value<T>>(std::pow(value->getData(), power),
                                        typename Value<T>::ChildrenType{value},
                                        std::format("^{}", power));
  ret->setBackward([=]() {
    value->getGrad() +=
        power * std::pow(value->getData(), power - 1) * ret->getGrad();
  });
  return ret;
}

template <typename T>
std::shared_ptr<Value<T>> operator/(typename std::shared_ptr<Value<T>> lhs,
                                    typename std::shared_ptr<Value<T>> rhs) {
  return lhs * (rhs ^ -1.0);
}

template <typename T>
std::shared_ptr<Value<T>> operator/(T const &lhs,
                                    typename std::shared_ptr<Value<T>> rhs) {
  return lhs * (rhs ^ -1.0);
}

template <typename T>
std::shared_ptr<Value<T>> operator/(typename std::shared_ptr<Value<T>> lhs,
                                    T const &rhs) {
  return lhs * (std::make_shared<Value<T>>(rhs) ^ -1);
}

template <typename T>
std::shared_ptr<Value<T>> tanh(typename std::shared_ptr<Value<T>> value) {
  auto x = value->getData();
  auto t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
  auto ret = std::make_shared<Value<T>>(
      t, typename Value<T>::ChildrenType{value}, "tanh");

  ret->setBackward(
      [=]() { value->getGrad() += (1.0 - t * t) * ret->getGrad(); });
  return ret;
}

template <typename T>
std::shared_ptr<Value<T>> exp(typename std::shared_ptr<Value<T>> value) {
  auto x = value->getData();
  auto ret = std::make_shared<Value<T>>(
      std::exp(x), typename Value<T>::ChildrenType{value}, "exp");

  ret->setBackward(
      [=]() { value->getGrad() += ret->getData() * ret->getGrad(); });
  return ret;
}

template <typename T>
std::ostream &operator<<(std::ostream &out,
                         typename std::shared_ptr<Value<T>> value) {
  out << std::format("Value(data={}, grad={})", value->getData(),
                     value->getGrad());
  return out;
}

template <typename T>
struct std::formatter<typename std::shared_ptr<Value<T>>>
    : std::formatter<std::string> {
  auto format(std::shared_ptr<Value<T>> value, format_context &ctx) const {
    return formatter<string>::format(std::format("Value(data={}, grad={})",
                                                 value->getData(),
                                                 value->getGrad()),
                                     ctx);
  }
};
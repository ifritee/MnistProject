#include "dense.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

namespace cpp_keras
{
  namespace cpp_layers
  {
    Dense::Dense(int units, EActivation activation) : AbstractLayer(),
      m_activation(activation)
    {
      m_outputLinks = units;
    }

    Dense::~Dense()
    {

    }

    Output Dense::compile(tensorflow::Scope &root, Output output)
    {
      auto & m_varsW = (*m_vars)["W" + m_layerNumber];
      auto & m_varsB = (*m_vars)["B" + m_layerNumber];
      auto & m_shapesW = (*m_shapes)["W" + m_layerNumber];
      auto & m_shapesB = (*m_shapes)["B" + m_layerNumber];
      auto & m_assignsW = (*m_assigns)["W" + m_layerNumber + "_assign"];
      auto & m_assignsB = (*m_assigns)["B" + m_layerNumber + "_assign"];

      TensorShape tensorShape = {m_inputLinks, m_outputLinks};
      auto DenseScope = root.NewSubScope("Dense" + m_layerNumber);
      m_varsW = Variable(DenseScope.WithOpName("W"), tensorShape, DT_FLOAT);  // Нужна оптимизация
      m_shapesW = tensorShape;
      m_assignsW = Assign(DenseScope.WithOpName("W_assign"), m_varsW, XavierInit(DenseScope, m_inputLinks, m_outputLinks));
      tensorShape = {m_outputLinks};
      m_varsB = Variable(DenseScope.WithOpName("B"), tensorShape, DT_FLOAT);
      m_shapesB = tensorShape;
      m_assignsB = Assign(DenseScope.WithOpName("B_assign"), m_varsB, Input::Initializer(0.f, tensorShape));
      auto dense = Add(DenseScope.WithOpName("Dense_b"), MatMul(DenseScope.WithOpName("Dense_w"), output, m_varsW), m_varsB);
      Output ReturnOutput;
      if(m_activation == ARelu_en) {
        ReturnOutput = Relu(DenseScope.WithOpName("Relu"), dense);
      } else if (m_activation == ASoftmax_en) {
        ReturnOutput = Softmax(DenseScope.WithOpName("Soft"), dense);
      }
      return ReturnOutput;
    }

    Input Dense::XavierInit(Scope & scope, int in_chan, int out_chan)
    {
      Tensor tensor(DT_INT64, {2});

      float std = sqrt(6.f / (in_chan + out_chan));
      auto v = tensor.vec<int64>();
      v(0) = in_chan;
      v(1) = out_chan;

      auto rand = RandomUniform(scope, tensor, DT_FLOAT);
      return Multiply(scope, Sub(scope, rand, 0.5f), std * 2.f);
    }
  }
}

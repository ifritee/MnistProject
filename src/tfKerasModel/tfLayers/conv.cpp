#include "conv.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

namespace cpp_keras
{
  namespace cpp_layers
  {

    Conv::Conv(int filterSide) : AbstractLayer(),
      m_filterSide(filterSide)
    {

    }

    cpp_keras::cpp_layers::Conv::~Conv()
    {

    }

    tensorflow::Output cpp_keras::cpp_layers::Conv::compile(tensorflow::Scope &root, tensorflow::Output output)
    {
      TensorShape sp({m_filterSide, m_filterSide, m_inputLinks, m_outputLinks});
      auto convScope = root.NewSubScope("Conv" + m_layerNumber);
      (*m_vars)["W" + m_layerNumber] = Variable(convScope.WithOpName("W"), sp, DT_FLOAT);
      (*m_shapes)["W" + m_layerNumber] = sp;
      (*m_assigns)["W" + m_layerNumber + "_assign"] = Assign(convScope.WithOpName("W_assign"),
        (*m_vars)["W"+m_layerNumber], XavierInit(convScope, m_inputLinks, m_outputLinks, m_filterSide));
      sp = {m_outputLinks};
      (*m_vars)["B"+m_layerNumber] = Variable(convScope.WithOpName("B"), sp, DT_FLOAT);
      (*m_shapes)["B"+m_layerNumber] = sp;
      (*m_assigns)["B"+m_layerNumber+"_assign"] = Assign(convScope.WithOpName("B_assign"), (*m_vars)["B"+m_layerNumber], Input::Initializer(0.f, sp));
      auto conv = Conv2D(convScope.WithOpName("Conv"), output, (*m_vars)["W"+m_layerNumber], {1, 1, 1, 1}, "SAME");
      auto bias = BiasAdd(convScope.WithOpName("Bias"), conv, (*m_vars)["B"+m_layerNumber]);
      auto relu = Relu(convScope.WithOpName("Relu"), bias);
      return MaxPool(convScope.WithOpName("Pool"), relu, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME");
    }

    Input Conv::XavierInit(Scope &scope, int in_chan, int out_chan, int filter_side)
    {
      float std;
      Tensor t;
      if(filter_side == 0)
      { //Dense
          std = sqrt(6.f/(in_chan+out_chan));
          Tensor ts(DT_INT64, {2});
          auto v = ts.vec<int64>();
          v(0) = in_chan;
          v(1) = out_chan;
          t = ts;
      }
      else
      { //Conv
          std = sqrt(6.f/(filter_side*filter_side*(in_chan+out_chan)));
          Tensor ts(DT_INT64, {4});
          auto v = ts.vec<int64>();
          v(0) = filter_side;
          v(1) = filter_side;
          v(2) = in_chan;
          v(3) = out_chan;
          t = ts;
      }
      auto rand = RandomUniform(scope, t, DT_FLOAT);
      return Multiply(scope, Sub(scope, rand, 0.5f), std*2.f);
    }
  }
}

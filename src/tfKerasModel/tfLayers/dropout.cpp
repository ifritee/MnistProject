#include "dropout.h"

using namespace tensorflow;
using namespace tensorflow::ops;

namespace cpp_keras
{
  namespace cpp_layers
  {
    DropOut::DropOut(float rate): AbstractLayer(),
      m_dropRateVar(rate)
    {

    }

    DropOut::~DropOut()
    {

    }

    tensorflow::Output DropOut::compile(tensorflow::Scope & root, tensorflow::Output output)
    {
      //----- Слой Drop -----
      const float skipDropVar = 0.f;
      auto DropScope = root.WithOpName("Dropout");
      auto rand = RandomUniform(DropScope, Shape(DropScope.WithOpName("DropoutShape"), output), DT_FLOAT);
      auto sub = Sub(DropScope.WithOpName("DropoutSub"), 1.f, m_dropRateVar);
      auto add = Add(DropScope.WithOpName("DropoutAdd"), sub, skipDropVar);
      auto add2 = Add(DropScope.WithOpName("DropoutAdd2"), rand, add);
      auto floor = Floor(DropScope.WithOpName("DropoutFloor"), add2);
      auto div = Div(DropScope.WithOpName("DropoutDiv"), output, m_dropRateVar);
      auto multi = Multiply(DropScope.WithOpName("DropoutMultiply"), div, floor);
      return multi;
    }
  }
}

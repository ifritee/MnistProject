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
      m_outputLinks = m_inputLinks;
      //----- Слой Drop -----
      const float skipDropVar = 0.f;
      auto DropScope = root.NewSubScope("Dropout" + m_layerNumber);
      auto rand = RandomUniform(DropScope, Shape(DropScope.WithOpName("Shape"), output), DT_FLOAT);
      auto sub = Sub(DropScope.WithOpName("Sub"), 1.f, m_dropRateVar);
      auto add = Add(DropScope.WithOpName("Add1"), sub, skipDropVar);
      auto add2 = Add(DropScope.WithOpName("Add2"), rand, add);
      auto floor = Floor(DropScope.WithOpName("Floor"), add2);
      auto div = Div(DropScope.WithOpName("Div"), output, m_dropRateVar);
      auto multi = Multiply(DropScope.WithOpName("Multiply"), div, floor);
      return multi;
    }
  }
}

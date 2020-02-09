#include "dense.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

namespace cpp_keras
{
  namespace cpp_layers
  {
    Dense::Dense(int units, EActivation activation) : AbstractLayer()
    {
      m_outputLinks = units;
    }

    Dense::~Dense()
    {

    }

    Output Dense::compile(tensorflow::Scope &root, Output output)
    {
      return Output();
    }
  }
}

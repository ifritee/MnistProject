#include "flatten.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

namespace cpp_keras
{
  namespace cpp_layers
  {
    Flatten::Flatten(vector<int> input_shape) : AbstractLayer()
    {
      for(auto value: input_shape) {
        m_length *= value;
      }
      m_inputLinks = m_length;
    }

    Flatten::~Flatten()
    {

    }

    Output Flatten::compile(tensorflow::Scope & root, Output output)
    {
      Scope flattenLayer = root.NewSubScope("FlattenLayer" + m_layerNumber);
      auto flat = Reshape(flattenLayer.WithOpName("Reshape"), output, {-1, m_inputLinks});
      m_outputLinks = m_length;
      return flat;
    }
  }
}

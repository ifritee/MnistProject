
#include <string>

#include "abstractlayer.h"

namespace cpp_keras
{
  namespace cpp_layers
  {

    static int layersCount = 0; ///< @brief Сквозная нумерация слоев

    AbstractLayer::AbstractLayer()
    {
      m_layerNumber = std::to_string(++layersCount);
    }

    AbstractLayer::~AbstractLayer()
    {

    }

    void AbstractLayer::setNetworkMaps(std::map<std::string, tensorflow::Output> *vars, std::map<std::string, tensorflow::TensorShape> *shapes, std::map<std::string, tensorflow::Output> *assigns)
    {
      m_vars = vars;
      m_shapes = shapes;
      m_assigns = assigns;
    }
  }
}

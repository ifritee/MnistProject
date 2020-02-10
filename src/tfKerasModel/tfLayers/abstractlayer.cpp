
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
  }
}

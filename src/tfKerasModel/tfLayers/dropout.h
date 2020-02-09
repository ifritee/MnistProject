#ifndef DROPOUT_H
#define DROPOUT_H

#include "abstractlayer.h"

namespace tensorflow {
  class Scope;
}

namespace cpp_keras
{
  namespace cpp_layers
  {
    class DropOut : public AbstractLayer
    {
      float m_dropRateVar;  ///< @brief
    public:
      DropOut(float);
      virtual ~DropOut();
      /** @brief Сборка очередного слоя выявление входов и выходов*/
      virtual tensorflow::Output compile(tensorflow::Scope & root, tensorflow::Output output) override;
    };
  }
}

#endif // DROPOUT_H

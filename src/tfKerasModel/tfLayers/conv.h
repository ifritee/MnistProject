#ifndef CONV_H
#define CONV_H

#include "../constants.h"
#include "abstractlayer.h"

namespace tensorflow {
  class Scope;
  class Input;
}

namespace cpp_keras
{
  namespace cpp_layers
  {
    class Conv : public AbstractLayer
    {
      int m_filterSide; ///< @brief Фильтрация

    public:
      Conv(int filterSide);
      /** @brief Деструктор */
      virtual ~Conv();
      /** @brief Сборка очередного слоя выявление входов и выходов*/
      virtual tensorflow::Output compile(tensorflow::Scope & root, tensorflow::Output output) override;

    private:
      tensorflow::Input XavierInit(tensorflow::Scope &scope, int in_chan, int out_chan, int filter_side);
    };
  }
}

#endif // CONV_H

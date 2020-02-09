#ifndef DENSE_H
#define DENSE_H

#include "../constants.h"
#include "abstractlayer.h"

namespace tensorflow {
  class Scope;
}

namespace cpp_keras
{
  namespace cpp_layers
  {
    /**
     * @brief Плотный слой
     */
    class Dense : public AbstractLayer
    {
      EActivation m_activation; ///< @brief Функция активации для этого слоя
    public:
      /**
       * @brief Dense Конструктор
       * @param units Количество нейронов в слое
       * @param activation Функция активации
       */
      Dense(int units, EActivation activation);
      /** @brief Деструктор */
      virtual ~Dense();
      /** @brief Сборка очередного слоя выявление входов и выходов*/
      virtual tensorflow::Output compile(tensorflow::Scope & root, tensorflow::Output output) override;
    };
  }
}

#endif // DENSE_H

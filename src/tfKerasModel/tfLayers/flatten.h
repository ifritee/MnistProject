#ifndef FLATTEN_H
#define FLATTEN_H

#include <vector>

#include "abstractlayer.h"

namespace tensorflow {
  class Scope;
}

namespace cpp_keras
{
  namespace cpp_layers
  {
    /**
     * @brief Flatten Слой выпрямления
     */
    class Flatten : public AbstractLayer
    {
      int m_length = 1;  ///< @brief общая длина

    public:
      /**
       * @brief Конструктор
       * @param input_shape параметр трансформации в одномерный вектор
       */
      Flatten(std::vector<int> input_shape);
      virtual ~Flatten();
      /** @brief Сборка очередного слоя выявление входов и выходов*/
      virtual tensorflow::Output compile(tensorflow::Scope & root, tensorflow::Output output) override;
    };
  }
}
#endif // FLATTEN_H

#ifndef ABSTRACTLAYER_H
#define ABSTRACTLAYER_H

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/summary/summary_file_writer.h"

namespace cpp_keras
{
  namespace cpp_layers
  {
    class AbstractLayer
    {
    public:
      AbstractLayer();
      virtual ~AbstractLayer();
      /** @brief Сборка очередного слоя выявление входов и выходов*/
      virtual tensorflow::Output compile(tensorflow::Scope & root, tensorflow::Output output) = 0;
      /** @brief Количество выходных связей */
      int outputLinks() { return m_outputLinks; }
      /** @brief Зададим количество входов */
      void setInputLinks(int qty) { m_inputLinks = qty; }

    protected:
      int m_inputLinks;  ///< @brief Количество входов
      int m_outputLinks; ///< @brief Количество выходов
    };
  }
}
#endif // ABSTRACTLAYER_H

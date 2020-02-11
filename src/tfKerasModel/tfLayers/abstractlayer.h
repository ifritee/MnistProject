#ifndef ABSTRACTLAYER_H
#define ABSTRACTLAYER_H

#include <map>
#include <string>

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
      /**
       * @brief setNetworkMaps Задание глобальных наборов данных сети
       * @param vars переменные сети
       * @param shapes Формы сети
       * @param assigns Назначения сети
       */
      void setNetworkMaps( std::map<std::string, tensorflow::Output> * vars,
                                  std::map<std::string, tensorflow::TensorShape> * shapes,
                                  std::map<std::string, tensorflow::Output> * assigns);

    protected:
      int m_inputLinks;  ///< @brief Количество входов
      int m_outputLinks; ///< @brief Количество выходов

      std::string m_layerNumber;  ///< @brief Номер слоя (сквозной)
      std::map<std::string, tensorflow::Output> * m_vars;
      std::map<std::string, tensorflow::TensorShape> * m_shapes;
      std::map<std::string, tensorflow::Output> * m_assigns;
    };
  }
}
#endif // ABSTRACTLAYER_H

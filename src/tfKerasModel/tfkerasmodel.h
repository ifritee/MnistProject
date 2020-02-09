#ifndef TFKERASMODEL_H
#define TFKERASMODEL_H

#include <list>
#include <string>

#include "constants.h"
#include "tfLayers/abstractlayer.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/summary/summary_file_writer.h"

/**
 * @namespace cpp_keras
 * Область видимости для модели Keras
 */
namespace cpp_keras
{
  /**
   * @class TFKerasModel
   * @brief Класс-представление Keras-модели на C++
   */
  class TFKerasModel
  {
    tensorflow::Scope m_root; ///< @brief Основной граф
    std::list<cpp_layers::AbstractLayer *> m_layers;  ///< @brief Все слои в модели
    EModelArchitect m_architecture; ///< @brief архитектура модели

  public:
    /**
     * @brief Основной конструктор
     * @param arch архитектура сети
     */
    TFKerasModel(EModelArchitect arch = MASequental);
    /** @brief Деструктор */
    virtual ~TFKerasModel();
    /**
     * @brief add Добавляет в модель слой
     * @param layer Слой
     */
    tensorflow::Status add(cpp_layers::AbstractLayer * layer);

    /**
     * @brief compile Настройка процесса обучения
     * @param optimizer определяет процедуру обучения
     * @param loss минимизируется в процессе обучения
     * @param metrics мониторинга обучения
     * @return Статус настройки обучения
     */
    tensorflow::Status compile(const std::string & optimizer, const std::string & loss, std::vector<std::string> metrics );

    /**
     * @brief fit Обучение модели
     * @param data Набор данных
     * @param label Набор меток
     * @param epochs Колич
     * @return
     */
    tensorflow::Status fit(tensorflow::Tensor & data, tensorflow::Tensor & label, uint32_t epochs);
  };

}

#endif // TFKERASMODEL_H

#ifndef TFKERASMODEL_H
#define TFKERASMODEL_H

#include <list>
#include <string>

#include "constants.h"
#include "tfLayers/abstractlayer.h"

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
    std::unique_ptr<tensorflow::ClientSession> m_session; ///< @brief Клиентская сессия
    tensorflow::Output m_data;
    tensorflow::Output drop_rate_var;
    tensorflow::Output skip_drop_var;
    tensorflow::Output m_inputLabels;
    tensorflow::Output outLossVariable;

    std::list<cpp_layers::AbstractLayer *> m_layers;  ///< @brief Все слои в модели
    EModelArchitect m_architecture; ///< @brief архитектура модели
    //Network maps
    std::map<std::string, tensorflow::Output> m_vars; ///< @brief Все переменные сети
    std::map<std::string, tensorflow::TensorShape> m_shapes;  ///< @brief Все формы сети
    std::map<std::string, tensorflow::Output> m_assigns;  ///< @brief Назначения сети

    tensorflow::Output m_outClassification; ///< @brief Выходно классификатор
    std::vector<tensorflow::Operation> m_outGrads;  ///< @brief Выходные градиенты

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
    tensorflow::Status compile(std::string optimizerFunction, std::string loss, std::vector<std::string> metrics );

    /**
     * @brief fit Обучение модели
     * @param data Набор данных
     * @param label Набор меток
     * @param epochs Колич
     * @return
     */
    tensorflow::Status fit(tensorflow::Tensor & data, tensorflow::Tensor & label, uint32_t epochs);

  private:
    /**
     * @brief Задание функции потерь
     * @param Тип функции потерь
     */
    std::vector<tensorflow::Output> lossyFunction(const std::string & lossy);

    /** @brief Функция оптимизации */
    void optimizerFunction(const std::string & optim, std::vector<tensorflow::Output> & gradient, float learning_rate);

    tensorflow::Status initializeGraph();
  };

}

#endif // TFKERASMODEL_H

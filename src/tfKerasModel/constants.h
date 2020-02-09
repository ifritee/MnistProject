#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace cpp_keras
{
  /** @enum EActivation
   * @brief Функции ктивации */
  enum EActivation {
    ARelu_en, ///< @brief Непрерывная дифференцируемость
    ASoftmax_en ///< @brief Обобщение логистической функции для многомерного случая
  };

  /** @enum EModelType
   * @brief Типы модели
   */
  enum EModelArchitect {
    MASequental,  ///< @brief Линейная сеть (последовательно расположенные слои)
    MAFunctional  ///< @brief Сеть произвольного напрвления
  };
}

#endif // CONSTANTS_H

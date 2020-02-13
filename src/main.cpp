#include <iostream>

#include "tensorflow/core/public/version.h"

#include "tfKerasModel/tfkerasmodel.h"
#include "tfKerasModel/imageprocessing.h"
#include "tfKerasModel/tfLayers/dense.h"
#include "tfKerasModel/tfLayers/flatten.h"
#include "tfKerasModel/tfLayers/dropout.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/summary/summary_file_writer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

using namespace cpp_keras;
using namespace cpp_keras::cpp_layers;

int main()
{
  cout << "Tensorflow version: " << TF_VERSION_STRING <<endl;
  cout << "Tensorflow compiler version: " << tf_compiler_version() <<endl;

  //----- Пример данных для ускорения -----
//  auto inputA = Tensor(DT_FLOAT, TensorShape{16, 4, 4});
//  auto inputB = Tensor(DT_FLOAT, TensorShape{4, 4});
//  std::vector<float> vecX1 = {1.f, 2.f, 3.f, 4.f};
//  std::vector<float> vecX2 = {5.f, 6.f, 7.f, 8.f};
//  auto dst1 = inputA.flat<float>().data();
//  auto dst2 = inputB.flat<float>().data();

//  for(int y = 0; y < 4; ++y) {
//    for(int i = 0; i < 4; ++i) {
//      copy_n(vecX1.begin(), vecX1.size(), dst1);
//      dst1 += 4;
//    }
//    copy_n(vecX2.begin(), vecX2.size(), dst2);
//    dst2 += 4;
//  }
  //-----------------------------------------

//  std::vector<std::pair<string, tensorflow::Tensor>> inputs = { {"X1", inputA},
//                                                                {"X2", inputB} };

  ImageProcessing imageProcessing;
  imageProcessing.loadMNISTDataset("data/mnist");

  TFKerasModel tfKerasModel;
  TF_CHECK_OK(tfKerasModel.add(new Flatten({28, 28})));
  TF_CHECK_OK(tfKerasModel.add(new Dense(128, ARelu_en)));
  TF_CHECK_OK(tfKerasModel.add(new DropOut(0.2)));
  TF_CHECK_OK(tfKerasModel.add(new Dense(10, ASoftmax_en)));

  TF_CHECK_OK(tfKerasModel.compile("adam", "sparse_categorical_crossentropy", {"accuracy"}));

//  TF_CHECK_OK(tfKerasModel.fit(inputA, inputB, 10));

  TF_CHECK_OK(tfKerasModel.fit(*imageProcessing.tensorDataTrain(), *imageProcessing.tensorLabelTrain(), 10));

//  std::cout<<trainLabel.DebugString(784)<<endl;

  return 0;
}

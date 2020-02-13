#include <arpa/inet.h>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/summary/summary_file_writer.h"

#include "imageprocessing.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

namespace cpp_keras {

  ImageProcessing::ImageProcessing()
  {

  }

  ImageProcessing::~ImageProcessing()
  {
    delete m_tensorDataTrain;
    delete m_tensorLabelTrain;
    delete m_tensorDataTest;
    delete m_tensorLabelTest;

//    m_dataTrain.clear();
//    m_labelTrain.clear();
//    m_dataTest.clear();
//    m_labelTest.clear();
  }

  tensorflow::Tensor * ImageProcessing::tensorDataTrain() const
  {
    return m_tensorDataTrain;
  }

  tensorflow::Tensor * ImageProcessing::tensorLabelTrain() const
  {
    return m_tensorLabelTrain;
  }

  tensorflow::Tensor *ImageProcessing::tensorDataTest() const
  {
    return m_tensorDataTest;
  }

  tensorflow::Tensor *ImageProcessing::tensorLabelTest() const
  {
    return m_tensorLabelTest;
  }

  void ImageProcessing::consoleOut(bool isTraining, int number)
  {
//    int count = 0, shift = 28;

//    TImagesData * images = isTraining ? &m_dataTrain : &m_dataTest;
//    std::vector<uint8_t> * labels = isTraining ? &m_labelTrain : &m_labelTest;
//    for(auto v: (*images)[number]) {
//      for(auto s: v) {
//        if(count++ % shift == 0) {
//          cout<<endl;
//        }
//        cout<< ((s > 0.7f) ? "#" : (s > 0.4f) ? "*" : (s > 0.1f) ? "." : " " );
//      }
//    }
//    cout<<endl<<"LABEL="<<(int)(*labels)[number]<<endl;
  }

  void ImageProcessing::loadMNISTDataset(const std::string &pathToMnistDB)
  {
    uint32_t num = 0, rows = 0, cols = 0/*, flatLength = 0*/;
    { //----- Сохраняем тренировочные данные в тензоре -----
      delete m_tensorDataTrain;
      TImagesData m_dataTrain = extractImages(pathToMnistDB + "/train-images-idx3-ubyte", num, rows, cols);
      m_tensorDataTrain = new Tensor(DT_FLOAT, TensorShape{static_cast<int>(m_dataTrain.size()), rows, cols});
      auto dst = m_tensorDataTrain->flat<float>().data();
      for(auto row: m_dataTrain) {
        for(auto col: row) {
          std::copy_n(col.begin(), cols, dst);
          dst += cols;
        }
      }
      //----- Сохраняем метки тренировочных данных в тензоре -----
      delete m_tensorLabelTrain;
      vector<float> m_labelTrain = extractLabels(pathToMnistDB + "/train-labels-idx1-ubyte");
      m_tensorLabelTrain = new Tensor(DT_FLOAT, TensorShape{static_cast<int>(m_labelTrain.size())});
      copy_n(m_labelTrain.begin(), m_labelTrain.size(), m_tensorLabelTrain->flat<float>().data());
    }
    { //----- Сохраняем тестовые данные в тензоре -----
      delete m_tensorDataTest;
      TImagesData m_dataTest = extractImages(pathToMnistDB + "/t10k-images-idx3-ubyte", num, rows, cols);
      m_tensorDataTest = new Tensor(DT_FLOAT, TensorShape{static_cast<int>(m_dataTest.size()), rows, cols});
      auto dst = m_tensorDataTest->flat<float>().data();
      for(auto row: m_dataTest) {
        for(auto col: row) {
          std::copy_n(col.begin(), cols, dst);
          dst += cols;
        }
      }

      //----- Сохраняем метки тестовых данных в тензоре -----
      delete m_tensorLabelTest;
      vector<float> m_labelTest = extractLabels(pathToMnistDB + "/t10k-labels-idx1-ubyte");
      m_tensorLabelTest = new Tensor(DT_FLOAT, TensorShape{static_cast<int>(m_labelTest.size())});
      copy_n(m_labelTest.begin(), m_labelTest.size(), m_tensorLabelTest->flat<float>().data());
    }
  }

  TImagesData ImageProcessing::extractImages(const std::string & file, uint32_t & num, uint32_t & rows, uint32_t & cols)
  {
    ifstream is(file);
    if (!is) {
      throw logic_error("can't open file: " + file);
    }
    uint32_t magic = readUint32(is);
    if (magic != 2051) {
      throw logic_error("bad magic: " + to_string(magic));
    }
    num = readUint32(is);
    rows = readUint32(is);
    cols = readUint32(is);
    TImagesData imagesData;
    while(num--) {  // Пройдем по всем элементам
      vector< vector< float > > image;
      for (uint32_t r = 0; r < rows; ++r) {
        vector< float > row;
        for(uint32_t c = 0; c < cols; ++c) {
          uint8_t byte = readUint8(is);
          row.push_back(static_cast<float>(byte)/255.f);
        }
        image.push_back(row);
      }
//      for(uint32_t i = 0; i < rows*cols; ++i) {
//        uint8_t byte = readUint8(is);
//        image.push_back(byte);
////        image.push_back(static_cast<float>(byte)/255.f);
//      }
      imagesData.push_back(image);
    }
    return imagesData;
  }

  std::vector<float> ImageProcessing::extractLabels(const string& file)
  {
    ifstream is(file);
    if (!is) {
      throw logic_error("can't open file: " + file);
    }
    uint32_t magic = readUint32(is);
    if (magic != 0x801) {
      throw logic_error("bad magic: " + to_string(magic));
    }
    uint32_t num = readUint32(is);
    vector<float> labels;
    for (size_t i = 0; i < num; ++i) {
      uint8_t byte = readUint8(is);
      labels.push_back(static_cast<float>(byte));
    }
    return labels;
  }

  uint8_t ImageProcessing::readUint8(ifstream & is)
  {
    uint8_t data = 0;
    auto read_count = is.readsome(reinterpret_cast<char*>(&data), 1);
    if (read_count != 1) {
      throw logic_error("can't read 1 byte");
    }
    return data;
  }

  uint32_t ImageProcessing::readUint32(ifstream & is)
  {
    uint32_t data = 0;
    auto read_count = is.readsome(reinterpret_cast<char*>(&data), 4);
    if (read_count != 4) {
      throw logic_error("can't read 4 bytes");
    }
    return ntohl(data);
  }
}

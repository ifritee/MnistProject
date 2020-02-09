
#include <iostream>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_file_writer.h"

#include "tfkerasmodel.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

namespace cpp_keras
{

  TFKerasModel::TFKerasModel(EModelArchitect arch) :
    m_root(Scope::NewRootScope()),
    m_architecture(arch)
  {

  }

  TFKerasModel::~TFKerasModel()
  {
    for(auto layer: m_layers) {
      delete layer;
    }
    m_layers.clear();
  }

  Status TFKerasModel::add(cpp_layers::AbstractLayer * layer)
  {
    m_layers.push_back(layer);
    return Status::OK();
  }

  Status TFKerasModel::compile(const string & /*optimizer*/, const string & /*loss*/, std::vector<string> /*metrics*/)
  {
    Output data = Placeholder(m_root.WithOpName("DATA"), DT_FLOAT);
    int outputLinks = 0;
    for(auto layer: m_layers) {
      if(outputLinks > 0) {
        layer->setInputLinks(outputLinks);
      }
      data = layer->compile(m_root, data);
      outputLinks = layer->outputLinks();
    }
    return m_root.status();
  }

  Status TFKerasModel::fit(Tensor & /*data*/, Tensor & /*label*/, uint32_t /*epochs*/)
  {

    return m_root.status();
  }

}

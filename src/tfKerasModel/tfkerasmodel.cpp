
#include <iostream>
#include <map>

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
    //----- Запись всех слоев в граф -----
    int outputLinks = 0;
    for(auto layer: m_layers) {
      layer->setNetworkMaps(&m_vars, &m_shapes, &m_assigns);
      if(outputLinks > 0) {
        layer->setInputLinks(outputLinks);
      }
      data = layer->compile(m_root, data);
      outputLinks = layer->outputLinks();
    }
    //----- Создание графа -----
    GraphDef graph;
    TF_CHECK_OK(m_root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session->Create(graph));
    std::cout << graph.DebugString() <<std::endl;
    return m_root.status();
  }

  Status TFKerasModel::fit(Tensor & /*data*/, Tensor & /*label*/, uint32_t /*epochs*/)
  {

    return m_root.status();
  }

}

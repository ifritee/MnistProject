
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

  Status TFKerasModel::compile(string optimizer, string loss, std::vector<string> /*metrics*/)
  {
    m_data = Placeholder(m_root.WithOpName("DATA"), DT_FLOAT);
    drop_rate_var = Placeholder(m_root.WithOpName("drop_rate"), DT_FLOAT);
    skip_drop_var = Placeholder(m_root.WithOpName("skip_drop"), DT_FLOAT);
    //----- Запись всех слоев в граф -----
    int outputLinks = 0;
    for(auto layer: m_layers) {
      layer->setNetworkMaps(&m_vars, &m_shapes, &m_assigns);
      if(outputLinks > 0) {
        layer->setInputLinks(outputLinks);
      }
      m_data = layer->compile(m_root, m_data);
      outputLinks = layer->outputLinks();
    }
    m_outClassification = Sigmoid(m_root.WithOpName("Output_Classes"), m_data);
    //----- Создание графа -----
    GraphDef graph;
    TF_CHECK_OK(m_root.ToGraphDef(&graph));
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_CHECK_OK(session->Create(graph));
    //----- Оптимизация -----
    ::transform(optimizer.begin(), optimizer.end(), optimizer.begin(), ::tolower);
    ::transform(loss.begin(), loss.end(), loss.begin(), ::tolower);
    try {
      vector<Output> gradOutputs = lossyFunction(loss);
      optimizerFunction(optimizer, gradOutputs, 0.0001f);
    } catch(std::logic_error & ex) {
      return errors::InvalidArgument(ex.what());
    }
    TF_CHECK_OK(initializeGraph());

    std::cout << graph.DebugString() <<std::endl;
    return m_root.status();
  }

  Status TFKerasModel::fit(Tensor & data, Tensor & label, uint32_t epochs)
  {
    if(!m_root.ok()) {
      return m_root.status();
    }

    float loss = 0.f;
    float accuracy_sum = 0.f;

    for(uint32_t epoch = 0; epoch < epochs; ++epoch) {
      vector<float> results;
      vector<Tensor> out_tensors;
      TF_CHECK_OK(m_session->Run( {{m_data, data}, {m_inputLabels, label}, {drop_rate_var, 0.5f}, {skip_drop_var, 0.f}},
      {outLossVariable, m_outClassification}, m_outGrads, &out_tensors));
      loss = out_tensors[0].scalar<float>()(0);
      auto mat1 = label.matrix<float>();
      auto mat2 = out_tensors[1].matrix<float>();
      for(int i = 0; i < mat1.dimension(0); i++) {
        results.push_back((fabs(mat2(i, 0) - mat1(i, 0)) > 0.5f)? 0 : 1);
      }

    }



    return Status::OK();
  }

  //-----------------------------------------------------------------------

  vector<Output> TFKerasModel::lossyFunction(const string & lossy)
  {
    m_inputLabels = Placeholder(m_root.WithOpName("inputL"), DT_FLOAT);
    Scope scope_loss = m_root.NewSubScope("Loss_scope");
    if(lossy == "sparse_categorical_crossentropy") {
      outLossVariable = Mean(scope_loss.WithOpName("Loss"), SquaredDifference(scope_loss, m_outClassification, m_inputLabels), {0});
    } else {
      throw(std::logic_error("Loss function is undefined -> " + lossy));
    }
    TF_CHECK_OK(scope_loss.status());
    vector<Output> weightsBiases;
    for(pair<string, Output> i: m_vars) {
      weightsBiases.push_back(i.second);
    }
    vector<Output> gradOutputs;
    TF_CHECK_OK(AddSymbolicGradients(m_root, {outLossVariable}, weightsBiases, &gradOutputs));
    return gradOutputs;
  }

  void TFKerasModel::optimizerFunction(const string & optim, std::vector<tensorflow::Output> & gradient, float learning_rate)
  {
    int index = 0;
    if(optim == "adam") {
      for(pair<string, Output> i: m_vars) {
        //Applying Adam
        string s_index = to_string(index);
        auto m_var = Variable(m_root, m_shapes[i.first], DT_FLOAT);
        auto v_var = Variable(m_root, m_shapes[i.first], DT_FLOAT);
        m_assigns["m_assign"+s_index] = Assign(m_root, m_var, Input::Initializer(0.f, m_shapes[i.first]));
        m_assigns["v_assign"+s_index] = Assign(m_root, v_var, Input::Initializer(0.f, m_shapes[i.first]));

        auto adam = ApplyAdam(m_root, i.second, m_var, v_var, 0.f, 0.f, learning_rate, 0.9f, 0.999f, 0.00000001f, {gradient[index]});
        m_outGrads.push_back(adam.operation);
        index++;
      }
    } else {
      throw(std::logic_error("Optimize function is undefined -> " + optim));
    }
  }

  Status TFKerasModel::initializeGraph()
  {
    if(!m_root.ok()) {
      return m_root.status();
    }

    vector<Output> opsToRun;
    for(pair<string, Output> i: m_assigns) {
      opsToRun.push_back(i.second);
    }
    m_session = unique_ptr<ClientSession>(new ClientSession(m_root));
    TF_CHECK_OK(m_session->Run(opsToRun, nullptr));
    /*
        GraphDef graph;
        TF_RETURN_IF_ERROR(t_root.ToGraphDef(&graph));
        SummaryWriterInterface* w;
        TF_CHECK_OK(CreateSummaryFileWriter(1, 0, "/Users/bennyfriedman/Code/TF2example/TF2example/graphs", ".cnn-graph", Env::Default(), &w));
        TF_CHECK_OK(w->WriteGraph(0, make_unique<GraphDef>(graph)));
        */
    return Status::OK();
  }

}

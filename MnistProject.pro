TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

TENSORFLOWINCLUDE = $${PWD}/../../third_party/include
TENSORFLOWLIB = $${PWD}/../../third_party/lib/tensorflow

SOURCES += \
        src/main.cpp \
        src/tfKerasModel/imageprocessing.cpp \
        src/tfKerasModel/tfLayers/abstractlayer.cpp \
        src/tfKerasModel/tfLayers/dense.cpp \
        src/tfKerasModel/tfLayers/dropout.cpp \
        src/tfKerasModel/tfLayers/flatten.cpp \
        src/tfKerasModel/tfkerasmodel.cpp

INCLUDEPATH += \
    src \
    src/tfKerasModel \
    src/tfKerasModel/tfLayers \
    $${TENSORFLOWINCLUDE} \
    $${TENSORFLOWINCLUDE}/bazel-genfiles \
    $${TENSORFLOWINCLUDE}/bazel-tensorflow/external/eigen_archive \
    $${TENSORFLOWINCLUDE}/bazel-tensorflow/external/com_google_absl \
    $${TENSORFLOWINCLUDE}/bazel-tensorflow/external/com_google_protobuf/src \
    $${TENSORFLOWINCLUDE}/bazel-tensorflow/bazel-out/k8-opt/bin

HEADERS += \
    src/tfKerasModel/constants.h \
    src/tfKerasModel/imageprocessing.h \
    src/tfKerasModel/tfLayers/abstractlayer.h \
    src/tfKerasModel/tfLayers/dense.h \
    src/tfKerasModel/tfLayers/dropout.h \
    src/tfKerasModel/tfLayers/flatten.h \
    src/tfKerasModel/tfkerasmodel.h

DEPENDPATH += \
    src \
    src/tfKerasModel \
    src/tfKerasModel/tfLayers

win32:CONFIG(release, debug|release): LIBS += -L$$TENSORFLOWLIB/release/ -ltensorflow_cc -ltensorflow_framework
else:win32:CONFIG(debug, debug|release): LIBS += -L$$TENSORFLOWLIB/debug/ -ltensorflow_cc -ltensorflow_framework
else:unix: LIBS += -L$$TENSORFLOWLIB/ -ltensorflow_cc -ltensorflow_framework

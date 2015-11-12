TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt+
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp


INCLUDEPATH += C:\Developpement\glm

SOURCES += main.cpp


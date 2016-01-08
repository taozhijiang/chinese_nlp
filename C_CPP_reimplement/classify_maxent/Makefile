CC = gcc
CFLAGS = -g -O2
CPPFLAGS = 
CXX = g++
CXXFLAGS = $(CFLAGS)
PACKAGE = classify
PACKAGE_NAME = $(PACKAGE)
PACKAGE_STRING = $(PACKAGE_NAME)1.0
PACKAGE_VERSION = 1.0
SHELL = /bin/sh
VERSION = 1.0
SUBDIRS = source
EXTRAFLAGS = -g -I./include -std=c++11 -DLOGGER_LEVEL=LL_WARN -rdynamic
OBJDIR = obj

vpath %.cpp $(SUBDIRS)
vpath %.hpp include

srcs =  $(filter-out main.cpp, $(notdir $(wildcard $(SUBDIRS)/*.cpp)))
objs = $(srcs:%.cpp=$(OBJDIR)/%.o)

all : $(PACKAGE)
.PHONY : all

$(PACKAGE) : $(objs) $(SUBDIRS)/main.cpp
	$(CXX) -c $(CCFLAGS) $(EXTRAFLAGS) $(SUBDIRS)/main.cpp -o $(OBJDIR)/main.o
	$(CXX) $(CCFLAGS) $(EXTRAFLAGS) $(objs) $(OBJDIR)/main.o -o $(PACKAGE) 

$(objs) : $(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) -MMD -c $(CCFLAGS) $(EXTRAFLAGS) $< -o $@ 

#check header for obj reconstruction
-include $(OBJDIR)/*.d

.PHONY : clean 
clean :	
	-rm -fr $(PACKAGE)
	-rm -fr $(OBJDIR)

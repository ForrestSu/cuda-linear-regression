#################################################
include ../../../Makefile.mk

INCLUDES := -I../../../common/inc -I./
ALL_CCFLAGS += -dc -rdc=true  
ALL_CCFLAGS += -DDOUBLE_PRECISION -DVERIFY #use double
LIBRARIES += -lpthread -lcublas -lcublas_device -lcudadevrt 
################################################################################
.PHONY:all clean run

CUFiles := $(wildcard *.cu)
CUObjs  := $(CUFiles:%.cu=%.o)
# Target rules
all: testspd testge

%.o: %.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

%.o: %.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

testspd: testspd.o invertspd.o 
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	
testge:	testge.o invertge.o 	
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
clean:
	rm -f *.o *.linkinfo testge testspd


clobber: clean
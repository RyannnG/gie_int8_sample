SHELL=/bin/sh
CUDAVERSION?=cuda-$(shell dpkg -l | grep cuda-core | sed -e "s/ \{1,\}/ /g" | cut -d ' ' -f 3 | cut -d '.' -f 1,2 | sed -e "s/-.*//g" | sort -n | tail -n 1)
TARGET?=$(shell uname -m)
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
export CUDAVERSION
export TARGET
export VERBOSE
#samples=sampleMNIST sampleMNISTAPI sampleGoogleNet giexec sampleINT8 getDigits
samples=sampleINT8
.PHONY: all clean help test test_debug test_release
all: 
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample);)

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample);)

test: test_debug test_release

test_debug:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_debug -C $(sample);)

test_release:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_release -C $(sample);)

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo "\t$(sample)";)
	$(AT)echo "\nCommands:"
	$(AT)echo "\tall - build all samples."
	$(AT)echo "\tclean - clean all samples."
	$(AT)echo "\ttest[_debug|_release] - test all/release/debug samples."
	$(AT)echo "\nVariables:"
	$(AT)echo "\tCUDAVERSION - Specify the version of cuda you want to use, defaults to the latest installed version of cuda."
	$(AT)echo "\tTARGET - Specify the target to build for."
	$(AT)echo "\tVERBOSE - Specify verbose output."


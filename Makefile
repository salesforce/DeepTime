ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))

build: .require-config
	python -m experiments.forecast --config_path=${config} build_experiment

build-all: .require-path
	for config in $(shell ls ${path}/*.gin); do \
      make build config=$$config; \
    done

run: .require-command
	bash -c "`cat ${ROOT}/${command}`"

.require-config:
ifndef config
	$(error config is required)
endif

.require-command:
ifndef command
	$(error command is required)
endif

.require-path:
ifndef path
	$(error path is required)
endif

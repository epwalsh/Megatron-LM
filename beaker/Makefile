BEAKER_USER := $(shell beaker account whoami --format=json | jq -r '.[0].name')
TIMESTAMP := $(shell date "+%Y%m%d%H%M%S")
WORKSPACE = ai2/google_benchmarks
IMAGE_NAME = megatron-lm

.PHONY : beaker-image
beaker-image :
	docker build -f Dockerfile -t $(IMAGE_NAME) .
	beaker image create $(IMAGE_NAME) --name $(IMAGE_NAME)-tmp --workspace $(WORKSPACE)
	beaker image rename $(BEAKER_USER)/$(IMAGE_NAME) $(IMAGE_NAME)-$(TIMESTAMP) >/dev/null 2>&1 || true
	beaker image rename $(BEAKER_USER)/$(IMAGE_NAME)-tmp $(IMAGE_NAME)

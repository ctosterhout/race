PROJECT=formula-trend
REGISTRY=ai.registry.trendmicro.com

## Build an image
## Create a tag with rank
## Push images with rank to Docker registry
auto:
	sudo docker build -t $(PROJECT):latest -f Dockerfile .
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank

## Build an image
## Create a tag with rank.1
## Push images with rank.1 to Docker registry
auto-1:
	sudo docker build -t $(PROJECT):latest -f Dockerfile .
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank.1
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank.1

## Build an image
## Create a tag with rank.2
## Push images with rank.2 to Docker registry
auto-2:
	sudo docker build -t $(PROJECT):latest -f Dockerfile .
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank.2
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank.2

## Build an image
## Create a tag with rank.3
## Push images with rank.3 to Docker registry
auto-3:
	sudo docker build -t $(PROJECT):latest -f Dockerfile .
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank.3
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank.3

# Build an image
build:
	sudo docker build -t $(PROJECT):latest -f Dockerfile .

## Run a new container
run:
	sudo docker run -it --rm -p 4567:4567 $(PROJECT):latest

## Create a tag with rank
tag-rank:
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank

## Create a tag with rank.1
tag-rank-1:
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank.1

## Create a tag with rank.2
tag-rank-2:
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank.2

## Create a tag with rank.3
tag-rank-3:
	sudo docker tag $(PROJECT):latest $(REGISTRY)/$(team)/$(PROJECT):rank.3

## Remove image which is tagged with rank
untag-rank:
	sudo docker rmi $(REGISTRY)/$(team)/$(PROJECT):rank

## Remove image which is tagged with rank.1
untag-rank-1:
	sudo docker rmi $(REGISTRY)/$(team)/$(PROJECT):rank.1

## Remove image which is tagged with rank.2
untag-rank-2:
	sudo docker rmi $(REGISTRY)/$(team)/$(PROJECT):rank.2

## Remove image which is tagged with rank.3
untag-rank-3:
	sudo docker rmi $(REGISTRY)/$(team)/$(PROJECT):rank.3

## Log in to a Docker registry
login:
	sudo docker login $(REGISTRY)

## Push an image with rank to Docker registry
push-rank:
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank

## Push an image with rank to Docker registry
push-rank-1:
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank.1

## Push an image with rank to Docker registry
push-rank-2:
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank.2

## Push an image with rank to Docker registry
push-rank-3:
	sudo docker push $(REGISTRY)/$(team)/$(PROJECT):rank.3

## Clean untagged images
clean:
	sudo docker rmi $$(sudo docker images -f "dangling=true" -q)
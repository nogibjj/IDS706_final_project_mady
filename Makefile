IMAGE_NAME = mady_final_project
DOCKER_ID_USER = ag826


install:
	pip install --upgrade pip  &&\
	pip install -r requirements.txt  &&\
	pip install --upgrade awscli botocore

format:
	black *.py #format all files	

lint:
	pylint --disable=R,C --ignore-patterns=test_.*?py *.py

test:
	python -m pytest -cov test.py

generate_and_push:
	git config --local user.email "action@github.com"
	git config --local user.name "GitHub Action"
	git pull
	git add .
	git commit -m "rerun push" --allow-empty
	git push
       
# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
	docker run -d -p 5001:5001 -e API_KEY=$(API_KEY) $(IMAGE_NAME)

# Remove the Docker image
clean:
	docker rmi $(IMAGE_NAME)

image_show:
	docker images

container_show:
	docker ps

push:
	docker login
	docker tag $(IMAGE_NAME) $(DOCKER_ID_USER)/$(IMAGE_NAME)
	docker push $(DOCKER_ID_USER)/$(IMAGE_NAME):latest

login:
	docker login -u ${DOCKER_ID_USER}


docker: build push

all: install format lint test generate_and_push


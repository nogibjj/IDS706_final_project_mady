 install:
	pip install --upgrade pip  &&\
	pip install -r requirements.txt

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
       
all: install format lint test generate_and_push
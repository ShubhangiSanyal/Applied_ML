# Assignment 4: Containerization & Continuous Integration

## containerization
- create a docker container for the flask app created in Assignment 3
- create a Dockerfile which contains the instructions to build the container, which include
  - installing the dependencies
  - copying app.py and score.py
  - launching the app by running “python app.py” upon entry<br>
  *(refer to file `Dockerfile`)*<br>
- build the docker image using Dockerfile<br>
  `docker build -t my_flask_app .`<br>
- run the docker container with appropriate port bindings<br>
  `docker run -p 5000:5000 my_flask_app`<br>
  *(One should now be able to access one's Flask app by navigating to http://localhost:5000 in your web browser.)*<br>
- in test.py write test_docker(..) function which does the following:
  - launches the docker container using commandline (e.g. os.sys(..), docker build and docker run)
  - sends a request to the localhost endpoint /score (e.g. using requests library) for a sample text
  - checks if the response is as expected
  - close the docker container

In coverage.txt, produce the coverage report using pytest for the tests in test.py

## continuous integration
Write a pre-commit git hook that will run the test.py automatically every time you try to commit the code to your local ‘main’ branch
copy and push this pre-commit git hook file to your git repo<br>

*(refer to `hooks` folder)*

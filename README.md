# cloud-functions

# Docker

Because this tensorflow requires a API wrapper, a Dockerfile was created. In order to use this dockerfile, you must build it, then run it

`docker build docker/ -t fastapi-tensorflow`    

NOTE: `-u $(id -u):$(id -g) ` cannot be applied here.  Something is up with Librosa that requires root level permissions.  
This may be a security fault.  Run this container with care.
```
docker run -d --rm --name fastapi-tensorflow \
	-p 8000:8000 \
	-v /Users/stu-personal/Desktop/github/cloud-functions:/home \
	-i fastapi-tensorflow
```

```
docker run --rm --name fastapi-tensorflow \
	-p 8000:8000 \
	-v /Users/stu-personal/Desktop/github/cloud-functions:/home \
	-i fastapi-tensorflow
```

To interact with this container
netstat -lntu
docker inspect -f '{{.State.Pid}}' fe7a60434459
docker exec -it fe7a60434459 netstat
```
docker exec -it fastapi-tensorflow /bin/bash
```

python3 unit_tests.py

To exit, simly type `exit` to return back to native shell.

# Errors

You may get this error when trying to run this container:

```
docker: Error response from daemon: Conflict. The container name "/flaskapp" is already in use by container 
```
do a `docker ps -a` then `docker stop <first three letters>` then  `docker rm <first three letters>`

then rerun the Flask container

```
docker run --name flaskapp --restart=always \
	-p 8181:80 \
	-v /Users/stu-personal/Desktop/github/cloud-functions:/home \
    -w /home/src \
	-d jazzdd/alpine-flask -d
```

Go here for a reference to Flask:

`https://flask.palletsprojects.com/en/1.1.x/quickstart/`

To view in browser:

`http://localhost:8181/instrument-classifier?instrument=trumpet`



FROM tensorflow/tensorflow:latest-py3

WORKDIR /home/src

RUN pip3 install fastapi[all] requests firebase-admin matplotlib pandas python_speech_features scipy keras sklearn tqdm librosa sox ffmpeg soundfile audioread pydub
RUN apt-get install sox libsndfile1 -y


EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]



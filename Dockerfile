# use Python 3.12 slim image as base image
FROM python:3.12-slim

# set the working directory inside the container
WORKDIR /app

# copy the file requirements.txt
COPY requirements.txt .

# install python libraries named in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy the whole project
COPY . .

# state that the app uses port 5000
EXPOSE 5000

# command executed when the docker start
CMD ["python", "app/app.py"]
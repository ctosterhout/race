FROM python:3.6.6-slim
  
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && apt-get clean
RUN pip install --trusted-host pypi.python.org --upgrade pip
RUN pip install --trusted-host pypi.python.org pillow flask-socketio eventlet numpy opencv-python tensorflow==1.9.0 keras==2.2.2

# Set the working directory to /app
WORKDIR /app

# Run main.py when the container launches, you should replace it with your program
ENTRYPOINT ["python", "main.py"] 

# Copy the current directory contents into the container at /app
RUN mkdir /log

COPY . /app
# Use an official Python runtime as the base image
FROM python:3.9.6

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y

# Install the Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clone the GitHub repository
# RUN git clone https://github.com/giangbioinformatics/smart-optical-sensor.git .

# Set the "bin" folder in the repository as a PATH
ENV PATH="smart-optical-sensor/smartsensor:${PATH}"
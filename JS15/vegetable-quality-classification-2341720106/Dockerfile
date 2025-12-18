# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV and related libraries
# libgl1 provides libGL.so.1; libglib2.0-0 and the X packages are commonly needed by opencv-python
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libgl1 \
		libglib2.0-0 \
		libsm6 \
		libxrender1 \
		libxext6 \
	&& rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
	# Ensure a compatible huggingface_hub version for Gradio (prevents HfFolder import error)
	&& pip install --no-cache-dir "huggingface_hub==0.13.3"

# Copy the application code and model into the container at /app
COPY app.py app.py
COPY model_mobilenetv2_classifier.keras model_mobilenetv2_classifier.keras

# Make port 7860 available to the world outside this container (Gradio default)
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]

# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /opt/app
WORKDIR /opt/app
COPY . /opt/app
# Install packages
RUN apt-get update -qq && apt-get install -y \
    pulseaudio \
    libasound2 libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev \
    python3-pyaudio \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Add user to 'pulse-access' group
RUN adduser root pulse-access

# Install Python packages
RUN pip install gdown 
# Copy requirements file and install Python dependencies
COPY requirements.txt /opt/app/
RUN pip install --no-cache-dir -r requirements.txt
#https://drive.google.com/file/d/1_BOj4zjAqz59EIRYvEiZphUScm9PKRo8/view?usp=sharing
# Download models using gdown
RUN gdown --id '1_BOj4zjAqz59EIRYvEiZphUScm9PKRo8' -O model_c_res2net.onnx
RUN gdown --id '1bb8DHuehwqy442PPW9tl8CR3Y6UH8UPm' -O model_nc_res2net.onnx

# Copy everything else
COPY . /opt/app

# Make port 80 available to the world outside this container
EXPOSE 80
RUN mkdir -p /opt/app/samples
# Define environment variables
ENV SAMPLE_RATE=16000
ENV NAME=World

# Set the entrypoint script
COPY entrypoint.sh /opt/bin/entrypoint.sh
ENTRYPOINT ["/opt/bin/entrypoint.sh"]

# Set the default command
CMD ["python3", "streaming_inference_with_display_onnx.py", "--model_chunked_path", "model_c_res2net.onnx", "--model_non_chunked_path", "model_nc_res2net.onnx", "--step_size", "3200", "--save_path", "/opt/app/samples/"]

# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Install build dependencies and Python3
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    pkg-config \
    libgtk-3-dev \
    libusb-1.0-0-dev \
    libx11-dev \
    freeglut3-dev \
    zlib1g-dev \
    liblapacke-dev \
    libopenblas-dev \
    libatlas-base-dev \
    python3 \
    python3-pip \
    udev \
    && rm -rf /var/lib/apt/lists/*


# Configure git to use HTTPS instead of SSH for GitHub URLs
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"

# Clone the libsurvive repository with its submodules
RUN git clone --recursive https://github.com/cntools/libsurvive.git /opt/libsurvive


WORKDIR /opt/libsurvive

# Copy udev rules for the HTC Vive into the container and reload them.
# (Note: In a container, you’ll likely need to run with --privileged and mount /dev for actual device access.)
RUN cp useful_files/81-vive.rules /etc/udev/rules.d/ 
#     && udevadm control --reload-rules && udevadm trigger

# Build libsurvive (this builds all tools and plugins)
RUN make

# Install the Python bindings (pysurvive)
RUN pip install scikit-build openvr scipy matplotlib && python3 setup.py install

# Copy your Python script (which reads tracker data) into the container.
# Make sure to name your script "app.py" (or adjust the COPY and CMD commands accordingly)
COPY test_tracker.py /opt/test_tracker.py

# Set the working directory for your application
WORKDIR /opt

# Default command: run your Python tracker script
CMD ["python3", "test_tracker.py"]

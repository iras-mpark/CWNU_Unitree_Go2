ARG BASE_IMAGE=dustynv/l4t-pytorch:r36.4.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

SHELL ["/bin/bash", "-c"]

# ------------------------------------------------------------
# 1. Basic tools + ROS2 apt repository
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    locales \
    software-properties-common \
    git \
    udev \
    usbutils \
    libusb-1.0-0 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && locale-gen en_US.UTF-8 \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. ROS2 + OAK dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-launch \
    ros-${ROS_DISTRO}-launch-ros \
    ros-${ROS_DISTRO}-rclpy \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-depthai-ros-driver \
    ros-${ROS_DISTRO}-depthai-ros-msgs \
    ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-pil \
    python3-requests \
    python3-tqdm \
    python3-psutil \
    python3-matplotlib \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 3. Python packages
#
# 중요:
# 베이스 이미지 pip index가 Jetson/NVIDIA 전용으로 잡혀 있으므로
# 일반 패키지는 PyPI index를 명시한다.
#
# 또 중요:
# 절대 pip install ultralytics 만 쓰지 않는다.
# torch/CUDA/nvidia-* 대형 wheel을 다시 받지 않기 위해 --no-deps 사용.
# ------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        py-cpuinfo \
        polars \
    && python3 -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --no-deps \
        ultralytics-thop \
    && python3 -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --no-deps \
        ultralytics==8.4.64

WORKDIR /ros2_ws

COPY src/oak_yolo_target /ros2_ws/src/oak_yolo_target
COPY startup.sh /startup.sh

RUN chmod +x /startup.sh && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install --packages-select oak_yolo_target

ENTRYPOINT ["/startup.sh"]

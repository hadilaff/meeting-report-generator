FROM python:3.11-slim

# Install system dependencies (git, gcc, etc.)
#RUN apk add --no-cache git ffmpeg build-base


RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libffi-dev \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# set work directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1



RUN pip install --upgrade pip
COPY ./requirements.txt /app
RUN pip install -r requirements.txt

# copy project
COPY . /app

EXPOSE 8000



# Make scripts executable
RUN chmod +x start.sh start-dev.sh


CMD ["./start.sh"]

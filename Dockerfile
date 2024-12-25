FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir

ENV HOME=/root/yunmengenvs
RUN mkdir -p $HOME
COPY . $HOME/

WORKDIR $HOME
EXPOSE 8000 5000

CMD ["python3", "app.py"]

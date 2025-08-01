# 使用预装了ubuntu22.04, CUDA 12.4.1 和 cuDNN 的镜像作为基础镜像
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 设置环境变量
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# 将项目代码复制到容器中
COPY . /app/yunmeng
WORKDIR /app/yunmeng
RUN chmod +x ./scripts/start.sh

# 暴露端口
EXPOSE 8002

# 更新软件源
RUN rm -rf /var/lib/apt/lists/* && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update    

# 安装 Python 3.10
RUN apt-get install --fix-missing -y python3 python3-pip
RUN python3 --version

# 安装 Python 依赖
RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install gunicorn gevent -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动命令
ENTRYPOINT ["./scripts/start.sh"]
# 默认行为：打印帮助
CMD ["--help"]

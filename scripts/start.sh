#!/bin/bash
set -e

case "$1" in
    web)
        # 启动 gunicorn,指定端口和进程数
        exec gunicorn -w ${GUNICORN_WORKERS:-5} \
                      -b 0.0.0.0:${PORT:-8002} \
                      app:app
        ;;
    task)
        # 启动离线任务，后面所有参数透传给 entry.py
        shift   # 去掉第一个参数 "task"
        exec python3 entry.py "$@"
        ;;
    *)
        echo "Usage:"
        echo "  docker run -e GUNICORN_WORKERS=3 -e PORT=8002 <image> web # gunicorn 启动 Web 服务"
        echo "  docker run <image> task --config xxx.json  # 启动本地任务"
        exit 1
        ;;
esac
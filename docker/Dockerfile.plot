ARG sourceimage=freqtradeorg/freqtrade
ARG sourcetag=develop
FROM ${sourceimage}:${sourcetag}

# 安装依赖
COPY requirements-plot.txt /freqtrade/

RUN pip install -r requirements-plot.txt --user --no-cache-dir
#!/bin/sh

# 使用BuildKit，否则在ARM架构上构建会失败
export DOCKER_BUILDKIT=1

# 镜像名称定义
镜像名称=freqtradeorg/freqtrade
缓存镜像=freqtradeorg/freqtrade_cache
GHCR镜像名称=ghcr.io/freqtrade/freqtrade

# 将分支名中的/替换为_以创建有效的标签
标签=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
标签_绘图=${标签}_plot
标签_FREQAI=${标签}_freqai
标签_FREQAI_RL=${标签_FREQAI}rl
标签_FREQAI_TORCH=${标签_FREQAI}torch
标签_PI="${标签}_pi"

标签_ARM=${标签}_arm
标签_绘图_ARM=${标签_绘图}_arm
标签_FREQAI_ARM=${标签_FREQAI}_arm
标签_FREQAI_RL_ARM=${标签_FREQAI_RL}_arm

echo "正在处理 ${标签}"

# 将提交信息添加到docker容器中
echo "${GITHUB_SHA}" > freqtrade_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "事件 ${GITHUB_EVENT_NAME}：完全重新构建 - 跳过缓存"
    # 构建常规镜像
    docker build -t freqtrade:${标签_ARM} .

else
    echo "事件 ${GITHUB_EVENT_NAME}：使用缓存构建"
    # 构建常规镜像
    docker pull ${镜像名称}:${标签_ARM}
    docker build --cache-from ${镜像名称}:${标签_ARM} -t freqtrade:${标签_ARM} .

fi

if [ $? -ne 0 ]; then
    echo "构建多架构镜像失败"
    return 1
fi

# 构建衍生镜像
docker build --build-arg sourceimage=freqtrade --build-arg sourcetag=${标签_ARM} -t freqtrade:${标签_绘图_ARM} -f docker/Dockerfile.plot .
docker build --build-arg sourceimage=freqtrade --build-arg sourcetag=${标签_ARM} -t freqtrade:${标签_FREQAI_ARM} -f docker/Dockerfile.freqai .
docker build --build-arg sourceimage=freqtrade --build-arg sourcetag=${标签_FREQAI_ARM} -t freqtrade:${标签_FREQAI_RL_ARM} -f docker/Dockerfile.freqai_rl .

# 为上传和下一步构建标记镜像
docker tag freqtrade:$标签_ARM ${缓存镜像}:$标签_ARM
docker tag freqtrade:$标签_绘图_ARM ${缓存镜像}:$标签_绘图_ARM
docker tag freqtrade:$标签_FREQAI_ARM ${缓存镜像}:$标签_FREQAI_ARM
docker tag freqtrade:$标签_FREQAI_RL_ARM ${缓存镜像}:$标签_FREQAI_RL_ARM

# 运行回测
docker run --rm -v $(pwd)/tests/testdata/config.tests.json:/freqtrade/config.json:ro -v $(pwd)/tests:/tests freqtrade:${标签_ARM} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy StrategyTestV3

if [ $? -ne 0 ]; then
    echo "回测运行失败"
    return 1
fi

# 显示镜像列表
docker images

# 推送镜像到仓库
docker push ${缓存镜像}:$标签_绘图_ARM
docker push ${缓存镜像}:$标签_FREQAI_ARM
docker push ${缓存镜像}:$标签_FREQAI_RL_ARM
docker push ${缓存镜像}:$标签_ARM

# 创建多架构镜像
# 确保所有包含的镜像都已推送到github，否则安装可能失败
echo "创建镜像清单"

docker manifest create ${镜像名称}:${标签} ${缓存镜像}:${标签} ${缓存镜像}:${标签_ARM} ${镜像名称}:${标签_PI}
docker manifest push -p ${镜像名称}:${标签}

docker manifest create ${镜像名称}:${标签_绘图} ${缓存镜像}:${标签_绘图} ${缓存镜像}:${标签_绘图_ARM}
docker manifest push -p ${镜像名称}:${标签_绘图}

docker manifest create ${镜像名称}:${标签_FREQAI} ${缓存镜像}:${标签_FREQAI} ${缓存镜像}:${标签_FREQAI_ARM}
docker manifest push -p ${镜像名称}:${标签_FREQAI}

docker manifest create ${镜像名称}:${标签_FREQAI_RL} ${缓存镜像}:${标签_FREQAI_RL} ${缓存镜像}:${标签_FREQAI_RL_ARM}
docker manifest push -p ${镜像名称}:${标签_FREQAI_RL}

# 创建特殊的Torch标签 - 与RL标签相同
docker manifest create ${镜像名称}:${标签_FREQAI_TORCH} ${缓存镜像}:${标签_FREQAI_RL} ${缓存镜像}:${标签_FREQAI_RL_ARM}
docker manifest push -p ${镜像名称}:${标签_FREQAI_TORCH}

# 复制镜像到ghcr.io
alias crane="docker run --rm -i -v $(pwd)/.crane:/home/nonroot/.docker/ gcr.io/go-containerregistry/crane"
mkdir .crane
chmod a+rwx .crane

echo "${GHCR_TOKEN}" | crane auth login ghcr.io -u "${GHCR_USERNAME}" --password-stdin

crane copy ${镜像名称}:${标签_FREQAI_RL} ${GHCR镜像名称}:${标签_FREQAI_RL}
crane copy ${镜像名称}:${标签_FREQAI_RL} ${GHCR镜像名称}:${标签_FREQAI_TORCH}
crane copy ${镜像名称}:${标签_FREQAI} ${GHCR镜像名称}:${标签_FREQAI}
crane copy ${镜像名称}:${标签_绘图} ${GHCR镜像名称}:${标签_绘图}
crane copy ${镜像名称}:${标签} ${GHCR镜像名称}:${标签}

# 对于develop构建，标记为latest
if [ "${标签}" = "develop" ]; then
    echo '将镜像标记为latest'
    docker manifest create ${镜像名称}:latest ${缓存镜像}:${标签_ARM} ${镜像名称}:${标签_PI} ${缓存镜像}:${标签}
    docker manifest push -p ${镜像名称}:latest

    crane copy ${镜像名称}:latest ${GHCR镜像名称}:latest
fi

# 显示镜像列表并清理
docker images
rm -rf .crane

# 从arm64节点清理旧镜像
docker image prune -a --force --filter "until=24h"
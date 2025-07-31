#!/bin/sh

# 以下内容假设已正确配置docker buildx环境

镜像名称=freqtradeorg/freqtrade
缓存镜像=freqtradeorg/freqtrade_cache
# 将分支名中的/替换为_以创建有效的标签
标签=$(echo "${BRANCH_NAME}" | sed -e "s/\//_/g")
标签_绘图=${标签}_plot
标签_FREQAI=${标签}_freqai
标签_FREQAI_RL=${标签_FREQAI}rl
标签_PI="${标签}_pi"

树莓派平台="linux/arm/v7"
echo "正在处理 ${标签}"
缓存标签=${缓存镜像}:${标签_PI}_cache

# 将提交信息添加到docker容器中
echo "${GITHUB_SHA}" > freqtrade_commit

if [ "${GITHUB_EVENT_NAME}" = "schedule" ]; then
    echo "事件 ${GITHUB_EVENT_NAME}：完全重新构建 - 跳过缓存"
    # 构建常规镜像
    docker build -t freqtrade:${标签} .
    # 构建树莓派镜像
    docker buildx build \
        --cache-to=type=registry,ref=${缓存标签} \
        -f docker/Dockerfile.armhf \
        --platform ${树莓派平台} \
        -t ${镜像名称}:${标签_PI} \
        --push \
        --provenance=false \
        .
else
    echo "事件 ${GITHUB_EVENT_NAME}：使用缓存构建"
    # 构建常规镜像
    docker pull ${镜像名称}:${标签}
    docker build --cache-from ${镜像名称}:${标签} -t freqtrade:${标签} .

    # 拉取上次构建以避免重新构建整个镜像
    # docker pull --platform ${树莓派平台} ${镜像名称}:${标签}
    # 禁用provenance以解决https://github.com/docker/buildx/issues/1509问题
    docker buildx build \
        --cache-from=type=registry,ref=${缓存标签} \
        --cache-to=type=registry,ref=${缓存标签} \
        -f docker/Dockerfile.armhf \
        --platform ${树莓派平台} \
        -t ${镜像名称}:${标签_PI} \
        --push \
        --provenance=false \
        .
fi

if [ $? -ne 0 ]; then
    echo "构建多架构镜像失败"
    return 1
fi

# 为上传和下一步构建标记镜像
docker tag freqtrade:$标签 ${缓存镜像}:$标签

# 构建衍生镜像
docker build --build-arg sourceimage=freqtrade --build-arg sourcetag=${标签} -t freqtrade:${标签_绘图} -f docker/Dockerfile.plot .
docker build --build-arg sourceimage=freqtrade --build-arg sourcetag=${标签} -t freqtrade:${标签_FREQAI} -f docker/Dockerfile.freqai .
docker build --build-arg sourceimage=freqtrade --build-arg sourcetag=${标签_FREQAI} -t freqtrade:${标签_FREQAI_RL} -f docker/Dockerfile.freqai_rl .

# 标记缓存镜像
docker tag freqtrade:$标签_绘图 ${缓存镜像}:$标签_绘图
docker tag freqtrade:$标签_FREQAI ${缓存镜像}:$标签_FREQAI
docker tag freqtrade:$标签_FREQAI_RL ${缓存镜像}:$标签_FREQAI_RL

# 运行回测
docker run --rm -v $(pwd)/tests/testdata/config.tests.json:/freqtrade/config.json:ro -v $(pwd)/tests:/tests freqtrade:${标签} backtesting --datadir /tests/testdata --strategy-path /tests/strategy/strats/ --strategy StrategyTestV3

if [ $? -ne 0 ]; then
    echo "回测运行失败"
    return 1
fi

# 显示镜像列表
docker images

# 推送镜像到缓存仓库
docker push ${缓存镜像}:$标签
docker push ${缓存镜像}:$标签_绘图
docker push ${缓存镜像}:$标签_FREQAI
docker push ${缓存镜像}:$标签_FREQAI_RL

# 显示镜像列表
docker images

if [ $? -ne 0 ]; then
    echo "镜像构建失败"
    return 1
fi
## CORS

整个部分仅在跨域情况下需要（例如，当您有多个机器人API运行在 `localhost:8081`、`localhost:8082` 等地址，并希望将它们整合到一个FreqUI实例中时）。

??? info "技术解释"
    所有基于Web的前端都受[CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)（跨域资源共享）的限制。
    由于大多数对Freqtrade API的请求都需要身份验证，因此正确的CORS策略是避免安全问题的关键。
    此外，标准禁止对带凭证的请求使用 `*` 通配符CORS策略，因此必须适当设置此配置。

用户可以通过 `CORS_origins` 配置项允许来自不同源URL访问机器人API。
它包含一个允许访问机器人API资源的URL列表。

假设您的应用部署在 `https://frequi.freqtrade.io/home/`，这意味着需要进行以下配置：
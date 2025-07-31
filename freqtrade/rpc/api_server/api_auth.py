import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.http import HTTPBasic, HTTPBasicCredentials

from freqtrade.rpc.api_server.api_schemas import AccessAndRefreshToken, AccessToken
from freqtrade.rpc.api_server.deps import get_api_config


logger = logging.getLogger(__name__)

ALGORITHM = "HS256"

router_login = APIRouter()


def verify_auth(api_config, username: str, password: str):
    """验证用户名/密码"""
    return secrets.compare_digest(username, api_config.get("username")) and secrets.compare_digest(
        password, api_config.get("password")
    )


httpbasic = HTTPBasic(auto_error=False)
security = HTTPBasic()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_user_from_token(token, secret_key: str, token_type: str = "access") -> str:  # noqa: S107
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("identity", {}).get("u")
        if username is None:
            raise credentials_exception
        if payload.get("type") != token_type:
            raise credentials_exception

    except jwt.PyJWTError:
        raise credentials_exception
    return username


# 这应该重新实现，以更好地与FastAPI提供的关于API令牌的现有工具保持一致
# https://github.com/tiangolo/fastapi/blob/master/fastapi/security/api_key.py
async def validate_ws_token(
    ws: WebSocket,
    ws_token: str | None = Query(default=None, alias="token"),
    api_config: dict[str, Any] = Depends(get_api_config),
):
    secret_ws_token = api_config.get("ws_token", None)
    secret_jwt_key = api_config.get("jwt_secret_key", "super-secret")

    # 检查ws_token是否在secret_ws_token中
    if ws_token and secret_ws_token:
        is_valid_ws_token = False
        if isinstance(secret_ws_token, str):
            is_valid_ws_token = secrets.compare_digest(secret_ws_token, ws_token)
        elif isinstance(secret_ws_token, list):
            is_valid_ws_token = any(
                [secrets.compare_digest(potential, ws_token) for potential in secret_ws_token]
            )

        if is_valid_ws_token:
            return ws_token

    # 检查ws_token是否为JWT
    try:
        user = get_user_from_token(ws_token, secret_jwt_key)
        return user
    # 如果令牌是JWT且有效，则返回用户
    except HTTPException:
        pass

    # 如果不匹配，关闭WebSocket连接
    await ws.close(code=status.WS_1008_POLICY_VIOLATION)


def create_token(data: dict, secret_key: str, token_type: str = "access") -> str:  # noqa: S107
    to_encode = data.copy()
    if token_type == "access":  # noqa: S105
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    elif token_type == "refresh":  # noqa: S105
        expire = datetime.now(timezone.utc) + timedelta(days=30)
    else:
        raise ValueError()
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": token_type,
        }
    )
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def http_basic_or_jwt_token(
    form_data: HTTPBasicCredentials = Depends(httpbasic),
    token: str = Depends(oauth2_scheme),
    api_config=Depends(get_api_config),
):
    if token:
        return get_user_from_token(token, api_config.get("jwt_secret_key", "super-secret"))
    elif form_data and verify_auth(api_config, form_data.username, form_data.password):
        return form_data.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="未授权",
    )


@router_login.post("/token/login", response_model=AccessAndRefreshToken)
def token_login(
    form_data: HTTPBasicCredentials = Depends(security), api_config=Depends(get_api_config)
):
    if verify_auth(api_config, form_data.username, form_data.password):
        token_data = {"identity": {"u": form_data.username}}
        access_token = create_token(
            token_data,
            api_config.get("jwt_secret_key", "super-secret"),
            token_type="access",  # noqa: S106
        )
        refresh_token = create_token(
            token_data,
            api_config.get("jwt_secret_key", "super-secret"),
            token_type="refresh",  # noqa: S106
        )
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确",
        )


@router_login.post("/token/refresh", response_model=AccessToken)
def token_refresh(token: str = Depends(oauth2_scheme), api_config=Depends(get_api_config)):
    # 刷新令牌
    u = get_user_from_token(token, api_config.get("jwt_secret_key", "super-secret"), "refresh")
    token_data = {"identity": {"u": u}}
    access_token = create_token(
        token_data,
        api_config.get("jwt_secret_key", "super-secret"),
        token_type="access",  # noqa: S106
    )
    return {"access_token": access_token}
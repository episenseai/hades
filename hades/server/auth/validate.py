from enum import Enum, unique
from typing import Optional, Union

from jose import jwt
from jose.exceptions import JOSEError, JWTError
from pydantic import UUID4, ValidationError

from ..logger import logger
from ..utils import ImmutBaseModel
from .env import get_jwt_config

# token issuer
TOKEN_ISS = "episense.ai"


@unique
class TokenType(str, Enum):
    ACCESS_TOKEN = "axx"
    REFRESH_TOKEN = "rxx"
    XACCESS_TOKEN = "xxx"


class DecodedToken(ImmutBaseModel):
    sub: UUID4
    scope: Union[str, list[str]]
    ttype: TokenType


async def validate_token(raw_token: str) -> Optional[DecodedToken]:
    config = get_jwt_config()

    try:
        decoded_token = jwt.decode(
            raw_token,
            config.get_secret_key("decode"),
            algorithms=config.authjwt_algorithm,
            options={
                "require_exp": True,
                "require_sub": True,
                "require_iss": True,
                "require_jti": False,
            },
            issuer=TOKEN_ISS,
        )
        logger.info(f"Decoded Bearer token for sub={decoded_token['sub']}")
        return DecodedToken(**decoded_token)
    except (JWTError, JOSEError) as exc:
        logger.error(f"JWT decode error: {exc}")
        return None
    except ValidationError as exc:
        logger.error(f"JWT token decode error: {exc}")
        return None
    except Exception:
        logger.exception("Unknown JWT decode error")
        return None

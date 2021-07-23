from enum import Enum, unique
from typing import Optional, Union

from jose import jwt
from jose.exceptions import JOSEError, JWTError, ExpiredSignatureError
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


async def validate_token(raw_token: str) -> tuple[Optional[DecodedToken], bool]:
    """
    Returns (decoded_token | None, is_expired)
    """
    config = get_jwt_config()

    try:
        decoded_token = jwt.decode(
            raw_token,
            # decoding only needs public key
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
        logger.info(f"decoded Bearer token: sub={decoded_token['sub']}")
        return (DecodedToken(**decoded_token), False)
    except ExpiredSignatureError:
        logger.info("got expired token")
        return (None, True)
    except (JWTError, JOSEError) as exc:
        logger.error(f"token decode: {exc}")
        return (None, False)
    except ValidationError as exc:
        logger.error(f"token validation: {exc}")
        return (None, False)
    except Exception:
        logger.exception("token decode: unexpected error:")
        return (None, False)

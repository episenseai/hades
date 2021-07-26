from datetime import timedelta
from functools import lru_cache
from typing import Optional

from pydantic import root_validator, validator
from pydantic.types import StrictBool, StrictStr

from ..utils import ImmutBaseModel
from ..env import env, Env
from ..logger import logger

# NOTE: Do not use this harccoded public/private key pair in production
public_key = """
-----BEGIN PUBLIC KEY-----
MIGeMA0GCSqGSIb3DQEBAQUAA4GMADCBiAKBgGBoQhqHdMU65aSBQVC/u9a6HMfK
A927aZOk7HA/kXuA5UU4Sl+UC9WjDhMQFk1PpqAjZdCqx9ajolTYnIfeaVHcLNpJ
Q6QXLnUyMnfwPmwYQ2rkuy5wI2NdO81CzJ/9S8MsPyMl2/CF9ZxM03eleE8RKFwX
CxZ/IoiqN4jVNjSrAgMBAAE=
-----END PUBLIC KEY-----
"""

symmetric_crypto = {"HS256", "HS384", "HS512"}
asymetric_crypto = {
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES521",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
    "EdDSA",
}

# same key as the one in titan
PRODUDCTION_PUB_KEY_PATH = "/etc/episense/pub/rsa_jwt_pub"


class JWTSettings(ImmutBaseModel):
    authjwt_algorithm: StrictStr = "RS512"
    authjwt_public_key: Optional[StrictStr] = public_key
    # private key not needed as are only validating the token
    authjwt_private_key: Optional[StrictStr] = None
    authjwt_access_token_expires: timedelta = timedelta(minutes=240)
    authjwt_refresh_token_expires: timedelta = timedelta(hours=8)
    authjwt_xaccess_token_expires: timedelta = timedelta(hours=1)
    authjwt_denylist_enabled: StrictBool = False
    authjwt_denylist_token_types: set[StrictStr] = {"access_token", "refresh_token"}

    @validator("authjwt_algorithm", pre=True, always=True)
    def algorithm(cls, _):
        """
        Always use RS512 assymetric crypto
        """
        return "RS512"

    @validator("authjwt_public_key", pre=True, always=True)
    def public_key_production(cls, key):
        """
        Always load public key from file mount at 'PRODUDCTION_PUB_KEY_PATH' in
        production
        """
        if env().ENV == Env.DEV:
            return key
        else:
            try:
                with open(PRODUDCTION_PUB_KEY_PATH) as public_key:
                    return public_key.read()
            except FileNotFoundError as ex:
                logger.error(f"missinng RSA public key file: {ex}")
                exit(1)
            except Exception as ex:
                logger.error(
                    f"occurred while processing RSA public key file ({PRODUDCTION_PUB_KEY_PATH}): {ex}"
                )
                exit(1)

    @root_validator(pre=False)
    def token_expires(cls, values):
        if values.get("authjwt_access_token_expires") >= values.get(
            "authjwt_refresh_token_expires"
        ):
            logger.error(
                "authjwt_access_token_expires must be less than authjwt_refresh_token_expires"
            )
            exit(1)
        return values

    def get_secret_key(self, method: Optional[str] = None):
        if self.authjwt_algorithm in asymetric_crypto:
            if method == "encode":
                return self.authjwt_private_key
            elif method == "decode":
                return self.authjwt_public_key
            else:
                logger.critical(f"unsupported_method = {method} for jwt")
                exit(1)
        elif self.authjwt_algorithm in symmetric_crypto:
            logger.critical(f"Symmetric crypto not supported = {self.authjwt_algorithm} for jwt")
            exit(1)
        else:
            logger.critical(f"Unsupported algorithm = {self.authjwt_algorithm} for jwt")
            exit(1)


@lru_cache
def get_jwt_config() -> JWTSettings:
    return JWTSettings()


get_jwt_config()

import datetime
import secrets
from logging import Logger
from typing import Optional, Tuple

import redis


class MetricsDB:
    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_password: Optional[str],
        redis_db: int,
        models_per_6hr: int,
        logger: Logger,
    ):
        self.redis = redis.Redis(
            host=redis_host, port=redis_port, password=redis_password, db=redis_db
        )
        self.models_per_6hr = models_per_6hr
        self.logger = logger

    def build_key(self, userid):
        return "models_build:{userid}".format(userid=userid)

    def cancel_key(self, userid):
        return "models_cancel:{userid}".format(userid=userid)

    def models_build(self, userid: str, requested: int = 1) -> Optional[Tuple[bool, int]]:
        if not isinstance(requested, int) or not userid:
            return None

        build_key = self.build_key(userid)
        cancel_key = self.cancel_key(userid)

        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
        current_timestamp = current_time.timestamp()
        previous_timestamp = (current_time - datetime.timedelta(hours=6)).timestamp()

        try:
            build_count = self.redis.zcount(build_key, previous_timestamp, current_timestamp)
            cancel_count = self.redis.zcount(cancel_key, previous_timestamp, current_timestamp)
            current = max(build_count - cancel_count, 0)

            if requested < 1:
                return (True, max(self.models_per_6hr - current, 0))
            if (current + requested) > self.models_per_6hr:
                self.logger.info(
                    f"QUOTA exceeded for models ({current=}, {requested=}, ({userid=})"
                )
                return (False, max(self.models_per_6hr - current, 0))

            events = {}
            for _ in range(requested):
                events[secrets.token_hex(16)] = current_timestamp

            if self.redis.zadd(build_key, events) == requested:
                return (True, max(self.models_per_6hr - current + requested, 0))
            self.logger.error(f"could not add models_build event (userid={userid})")
            return None
        except Exception as ex:
            self.logger.error(f"models_build event (userid={userid}): {ex}")
            return None

    def models_cancel(self, userid: str, requested: int = 1) -> Optional[int]:
        if not isinstance(requested, int) or not userid:
            return None
        if requested < 1:
            return 0

        cancel_key = self.cancel_key(userid)

        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
        current_timestamp = current_time.timestamp()

        try:
            events = {}
            for _ in range(requested):
                events[secrets.token_hex(16)] = current_timestamp

            if self.redis.zadd(cancel_key, events) == requested:
                return requested
            self.logger.error(f"could not add models_cancel event (userid={userid})")
            return None
        except Exception as ex:
            self.logger.error(f"models_cancel event (userid={userid}): {ex}")
            return None


if __name__ == "__main__":
    import logging

    logger = logging.getLogger("hades")
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

    metrics_db = MetricsDB(
        redis_host="localhost",
        redis_port=6379,
        redis_password=None,
        redis_db=4,
        models_per_6hr=10,
        logger=logger,
    )

    print(metrics_db.models_build("123456", requested=4))

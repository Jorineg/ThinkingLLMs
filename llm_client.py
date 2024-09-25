# scheduler for llm requests to openrouter
# when first request arrives, check for rate limit, add to queue
# while queue is not empty, check for rate limit, send request
# if request hits rate limit error, add again to queue and check current rate limit
#  if rate limit is lower than before, update rate limit
#  else decrease rate limit by 1

import asyncio
import requests
from collections import deque
from time import time
import dotenv
import os
from base_client import AsyncClient
import threading
import logging

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


class RateLimitError(Exception):
    pass


class TimeoutError(Exception):
    pass


class AsyncOpenRouterClient(AsyncClient):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.sync_session = requests.Session()
        self.request_times = deque()
        self.rate_limit = 10
        self.rate_period = 10
        self.update_rate_limit()

    # returns True if rate limit was decreased
    def update_rate_limit(self):
        url = "https://openrouter.ai/api/v1/auth/key"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        old_rate = self.rate_limit / self.rate_period
        response = self.sync_session.get(url, headers=headers)
        # print(response)
        result = response.json()
        self.rate_limit = result["data"]["rate_limit"]["requests"]
        self.rate_period = int(result["data"]["rate_limit"]["interval"][:-1])
        new_rate = self.rate_limit / self.rate_period
        return new_rate < old_rate

    async def stop(self):
        super().stop()
        self.sync_session.close()

    async def wait_for_rate_limit(self):
        now = time()
        self.request_times.append(now)

        oldest = None
        while len(self.request_times) > self.rate_limit:
            oldest = self.request_times.popleft()
        if oldest is not None:
            sleep_time = self.rate_period - (now - oldest)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def handle_request(self, id, data, callback):
        # print(data["messages"][-1]["content"])
        try:
            await self._make_openrouter_request(id, data, callback)
        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            logger.info("Retrying request...")
            rate_limit_decreased = self.update_rate_limit()
            if not rate_limit_decreased:
                logger.info("Decreasing rate limit by 1")
                self.rate_limit -= 1
                await asyncio.sleep(1)
            else:
                logger.info("Rate limit decreased, updating rate limit")
            await self.make_request(id, data, callback, put_first=True)
        except (TimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"Timeout error: {e}")
            logger.info("Retrying request...")
            await self.make_request(id, data, callback, put_first=True)
        except Exception as e:
            logger.error(f"Error in handle_request: {e}", exc_info=True)
        finally:
            self.queue.task_done()

    async def _make_openrouter_request(self, id, data, callback):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        logger.info(f"Id {id}: Making request to OpenRouter")
        try:
            async with self.session.post(url, headers=headers, json=data) as response:
                try:
                    result = await response.json()

                    openrouter_status = 200
                    if "error" in result:
                        logger.error(
                            f"Id {id}: Error in _make_openrouter_request: {result}"
                        )
                        openrouter_status = result["error"]["code"]

                    if openrouter_status == 429:
                        raise RateLimitError(result)
                    elif openrouter_status == 408 or openrouter_status == 502:
                        raise TimeoutError(result)
                    await callback(id, result)
                    # run callback in separate thread
                    # threading.Thread(target=callback, args=(id, result)).start()
                except Exception as e:
                    logger.error(
                        f"Id {id}: Error in _make_openrouter_request: {e}",
                        exc_info=True,
                    )
                finally:
                    logger.info(f"Id {id}: Received response from OpenRouter")
                if response.status != 200:
                    logger.error(
                        f"Id {id}: AsyncClient: Received response from OpenRouter, status {response.status}"
                    )
        except asyncio.TimeoutError as e:
            logger.error(f"Id {id}: Timeout error: {e}")
            logger.info(f"Id {id}: Restarting session")
            print("Restarting session")
            await self.restart_session()
            raise e

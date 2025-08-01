"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Simple backoff script.

Based on 
https://platform.openai.com/docs/guides/rate-limits/error-mitigation
"""

import random
import time
import openai

from loguru import logger


def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int | None = None,
    max_time: float | None = None,
    errors: tuple = (openai.RateLimitError,),
) -> callable:
    """Retry a function with exponential backoff.

    Arguments:
        initial_delay: When we hit the first error, amount of retry delay we start with.
        exponential_base: Used to compute delay.
        jitter: If True, add randomness to delay.
        max_retries: If set, raise after number of retries.
        max_time: If set, raise after time elapses in seconds.
        errors: Errors to catch for retries. Others will be propagated.

    Returns:
        Function with retries.

    Raises:
        Exception: if:
            - Number of retries is exceeded.
            - Time spent is exceeded.
            - Exception that is not expected.
    """

    def wrap(func):
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay
            start_time = time.time()

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specific errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if max_retries is not None:
                        if num_retries > max_retries:
                            raise Exception(
                                f"Maximum number of retries ({max_retries}) exceeded."
                            )
                    if max_time is not None:
                        if time.time() - start_time > max_time:
                            raise Exception(
                                f"Maximum time for retries ({max_time}) exceeded. Failed with errors {e}."
                            )

                    logger.warning(f"Error calling API ({e}). Retrying...")

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return wrap

from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import asyncio
import itertools
import copy
import time
try:
    import requests
    from aiohttp import ClientSession, TCPConnector
    from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
    from tqdm import tqdm
except ModuleNotFoundError:
    pass


from lm_eval.api.registry import register_model
from lm_eval.models.api_models import LogLikelihoodInputs, JsonChatStr
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.utils import eval_logger
from lm_eval.models.utils import Collator

@register_model("friendli-completions")
class FriendliCompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, tokenized_request=True, **kwargs
        )
        assert self._batch_size == 1, "If you want to use large batch, just increase `num_concurrent` value in model_args."

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        assert (
            self.tokenizer is not None
        ), "Tokenizer is required for loglikelihood tasks to compute context lengths."
        res = []

        def _collate(req: LogLikelihoodInputs):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=None,
        )
        # if concurrent then we'll batch in the async context
        chunked = re_ord.get_batched(n=self._batch_size if self._concurrent <= 1 else 0)
        if self._concurrent <= 1:
            pbar = tqdm(desc="Requesting API", total=len(requests))
            for chunk in chunked:
                # NOTE: inputs : List of (context tokens + continuation tokens)
                inputs, ctxlens, cache_keys = self.batch_logliklehood_requests([chunk])

                outputs = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=1, max=10),
                    reraise=True,
                )(self.model_call)(messages=inputs, generate=False, ctxlens=ctxlens)
                if isinstance(outputs, dict):
                    outputs = [outputs]
                for answer_, cache_key in zip(
                    self.parse_logprobs(
                        outputs=outputs, tokens=inputs, ctxlens=ctxlens
                    ),
                    cache_keys,
                ):
                    if answer_ is not None:
                        res.append(answer_)
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial(
                                "loglikelihood", cache_key, answer_
                            )
                        pbar.update(1)
        else:
            inputs, ctxlens, cache_keys = self.batch_logliklehood_requests(chunked)
            res = itertools.chain.from_iterable(
                asyncio.run(
                    self.get_batched_requests(
                        inputs, cache_keys, generate=False, ctxlens=ctxlens
                    )
                )
            )

        return re_ord.get_original(res)

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        try:
            response = requests.post(
                self.base_url,
                json=self._create_payload(
                    self.create_message(messages),
                    generate=generate,
                    gen_kwargs=gen_kwargs,
                    seed=self._seed,
                    **kwargs,
                ),
                headers=self.header,
            )
            if not response.ok:
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
            response.raise_for_status()
            return response.json()
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    async def amodel_call(
        self,
        session: ClientSession,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        cache_keys: list = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Union[List[str], List[Tuple[float, bool]], None]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            ctxlens=ctxlens,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(
                        f"API request failed with error message: {error_text}. Retrying..."
                    )
                # raising exception will retry the request
                response.raise_for_status()
                outputs = await response.json()
            answers = (
                self.parse_generations(
                    outputs=outputs,
                )
                if generate
                else self.parse_logprobs(
                    outputs=outputs,
                    tokens=messages,
                    ctxlens=ctxlens,
                )
            )
            if cache_keys:
                for res, cache in zip(answers, cache_keys):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        # If the retries also fail
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None
        except asyncio.TimeoutError:
            return
    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        **kwargs,
    ) -> dict:
        if generate:
            kwargs = copy.deepcopy(gen_kwargs) # edge case for num_concurrents > 1
            kwargs.pop("do_sample", False)
            max_tokens = kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = kwargs.pop("temperature", 0)
            eos = self.tokenizer.decode(self.eot_token_id)
            stop = kwargs.pop("until", [eos])
            if self.tokenized_requests:
                return {
                    "tokens": messages,
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "seed": seed,
                    **kwargs,
                }
            else:
                return {
                    "prompt": messages,
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "seed": seed,
                    **gen_kwargs,
                }
        else:
            # NOTE: when running non-generate task, we always use tokenized request
            ctxlens = kwargs.pop("ctxlens", None)
            assert ctxlens is not None
            forced_output_tokens_list = []
            input_list = []
            for message, ctxlen in zip(messages, ctxlens):
                input_list.append(message[:ctxlen])
                forced_output_tokens_list.append(message[ctxlen:])
            return {
                "model": self.model,
                "tokens": input_list,
                "temperature": 0,
                "include_output_logprobs": True,
                "forced_output_tokens": forced_output_tokens_list,
                "seed": seed,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(out["choices"], ctxlens):
                assert ctxlen > 0, "Context length must be greater than 0"
                forced_logprobs = sum(choice["logprobs"])
                res.append((forced_logprobs, False)) #NOTE: Can't use exact_match metrics when running non-generation task.
        return res

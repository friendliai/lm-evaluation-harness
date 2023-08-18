import requests
import transformers
import asyncio
import aiohttp
import uvloop
from tqdm.asyncio import tqdm_asyncio

from tqdm import tqdm
from lm_eval.base import BaseLM
from lm_eval import utils

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def check_health(session, master_uri):
    try:
        async with session.get(f"{master_uri}/health") as response:
            return response.status == 200
    except:
        return False
    
async def orca_async_completion(session, inps, req_url, forced_output_tokens=None,):
    data={}
    headers={}
    data["tokens"] = inps
    data["include_output_logprobs"] = "true"
    data["seed"] = 42
    data["forced_output_tokens"] = forced_output_tokens
    
    headers["Content-Type"]="application/json"
    async with session.post(req_url, headers=headers, json=data) as response:
        pred = await response.json()

    return pred["choices"][0]["logprobs"]

def orca_completion(inps, req_url, forced_output_tokens=None,):
    data={}
    headers={}
    data["tokens"] = inps
    data["include_output_logprobs"] = "true"
    data["seed"] = 42
    data["forced_output_tokens"] = forced_output_tokens
    
    headers["Content-Type"]="application/json"
    pred = requests.post(req_url, headers=headers, data=str(data)).json()
    return pred["choices"][0]["logprobs"]


class ORCALM(BaseLM):

    def __init__(self, model_name_or_path:str, req_url):
        super().__init__()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.req_url = req_url

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        raise NotImplementedError()

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)
        for _, context_enc, continuation_enc in tqdm(
            re_ord.get_reordered(),
            disable=disable_tqdm,
        ):
            total_len = len(context_enc)+len(continuation_enc)
            over_len = total_len-self.max_length
            inp = context_enc[over_len:] if over_len > 0 else context_enc
            
            forced_logprob = orca_completion(
                inps=inp,
                forced_output_tokens=continuation_enc,
                req_url=self.req_url
            )

            answer = (sum(forced_logprob), False) # Only Support metric calculating from logprobs ex) MultiChoiceTask
            res.append(answer)

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        # Not implement for generation task
        raise NotImplementedError()

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()


class ORCAASYNCLM(BaseLM):

    def __init__(self, model_name_or_path:str, req_url):
        super().__init__()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.req_url = req_url

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    async def _loglikelihood_tokens_async(self, re_ord, disable_tqdm=False):
        res = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as session:
            forced_logprobs = []
            for _, context_enc, continuation_enc in re_ord.get_reordered():
                total_len = len(context_enc)+len(continuation_enc)
                over_len = total_len-self.max_length
                inp = context_enc[over_len:] if over_len > 0 else context_enc
                
                forced_logprobs.append(asyncio.create_task(
                    orca_async_completion(
                        session=session,
                        inps=inp,
                        forced_output_tokens=continuation_enc,
                        req_url=self.req_url
                    )
                ))
            forced_logprobs = await tqdm_asyncio.gather(*forced_logprobs)
            for forced_logprob in forced_logprobs:
                answer = (sum(forced_logprob), False) # Only Support metric calculating from logprobs ex) MultiChoiceTask
                res.append(answer)

            return res

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)
        re_ord = utils.Reorderer(requests, _collate)

        loop=asyncio.get_event_loop()
        res = loop.run_until_complete(self._loglikelihood_tokens_async(re_ord, disable_tqdm))

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        # Not implement for generation task
        raise NotImplementedError()

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()

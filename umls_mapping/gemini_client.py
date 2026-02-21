import asyncio
import math
import nest_asyncio
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
from google import genai

class GeminiEmbedder:
    def __init__(self, api_key, model_name="text-embedding-004", rpm=150):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # Rate Limiting
        self.rps_limit = math.floor(rpm / 60)
        self.concurrency = math.ceil(rpm / 60)
        self.per_second_limiter = AsyncLimiter(self.rps_limit, 1.0)
        self.per_minute_limiter = AsyncLimiter(rpm, 60.0)
        self.semaphore = asyncio.Semaphore(self.concurrency)
        
        nest_asyncio.apply()

    async def _embed_batch(self, texts):
        retries = 5
        delay = 2
        while retries > 0:
            try:
                async with self.per_second_limiter, self.per_minute_limiter, self.semaphore:
                    response = await self.client.aio.models.embed_content(
                        model=self.model_name,
                        contents=texts,
                    )
                # Tiny sleep to smooth bursts
                await asyncio.sleep(1.0 / self.rps_limit)
                return [emb.values for emb in response.embeddings]
            except Exception as e:
                print(f"Error: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2
                retries -= 1
        return [None] * len(texts)

    async def embed_texts(self, text_list, batch_size=100):
        """Main entry point to embed a list of strings."""
        batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
        
        tasks = [self._embed_batch(batch) for batch in batches]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching Embeddings")
        
        # Flatten results
        #flat_embeddings = [emb for batch in results for emb in batch]
        final_embeddings = []
        for i, batch_res in enumerate(results):
            batch_len = len(batches[i])
            if batch_res:
                final_embeddings.extend(batch_res)
                # Pad if partial result (rare safety filter case)
                if len(batch_res) < batch_len:
                    final_embeddings.extend([None] * (batch_len - len(batch_res)))
            else:
                # Full batch failure
                final_embeddings.extend([None] * batch_len)
        return final_embeddings
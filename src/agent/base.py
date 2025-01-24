import os
import torch
import openai
import transformers

class BaseAgent:
    def __init__(self, model_name="OpenAI/gpt-4o-mini", cache_dir="../huggingface/hub", model_dtype=torch.bfloat16, api_key=None, model=None):
        self.model_name = model_name
        self.client = None
        self.model = None
        if "OpenAI" in self.model_name:
            self.model_name = self.model_name.strip('/').split('/')[-1]
            self.client = openai.OpenAI(
                api_key=openai.api_key or os.environ["OPENAI_API_KEY"] or api_key,
            )
        else:
            self.model = model or transformers.pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=model_dtype,
                device_map="auto",
                model_kwargs={"cache_dir":cache_dir},
            )
    
    def generate_hypothesis(self, background):
        raise NotImplementedError
    
    def call_llm(
        self, 
        messages: list[dict[str, str]], 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_return_sequences: int = 1,
        **kwargs
    ):
        if self.model is not None:
            if temperature > 0.0:
                response = self.model(messages, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, temperature=temperature, pad_token_id=self.model.tokenizer.eos_token_id, **kwargs)
                outputs = [response[i]["generated_text"][-1]["content"] for i in range(num_return_sequences)]
            else:
                response = self.model(messages, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.model.tokenizer.eos_token_id, **kwargs)
                outputs = [response[0]["generated_text"][-1]["content"]]
        else:
            response = self.client.chat.completions.create(
                model= self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=num_return_sequences if temperature > 0.0 else 1,
                **kwargs
            )
            outputs = [response.choices[i].message.content for i in range(num_return_sequences if temperature > 0.0 else 1)]
        return outputs
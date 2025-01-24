import os
import torch
import openai
import transformers
from liquid import Template
from src.agent.base import BaseAgent

class CoTAgent(BaseAgent):
    def __init__(self, model_name="OpenAI/gpt-4o-mini", cache_dir="../huggingface/hub", model_dtype=torch.bfloat16, api_key=None):
        super().__init__(model_name, cache_dir, model_dtype, api_key)
        self.system_prompt = "You are a scientist. Your task is to generate a scientific hypothesis following given instructions."
        self.user_template = Template('''### User Input
{{background}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
- Think step-by-step to derive the hypothesis.

2. **### Structured Output**:
- Present your proposed hypothesis in the following JSON format:
  ```json
  {
      "proposed_hypothesis": "Statement of the proposed hypothesis"
  }
  ```''')

    def generate_hypothesis(self, background, temperature=0.0, n_hypotheses=1, max_new_tokens=2048, **kwargs):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.user_template.render(background=background)
            }
        ]
        return self.call_llm(messages, temperature=temperature, num_return_sequences=n_hypotheses, max_new_tokens=max_new_tokens)
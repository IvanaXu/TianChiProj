from easyinstruct import BasePrompt, IEPrompt, ZeroshotCoTPrompt, FewshotCoTPrompt, BatchPrompt
from easyinstruct.utils.api import set_openai_key, set_anthropic_key, set_proxy

set_openai_key("")
set_anthropic_key("")
set_proxy("http://127.0.0.1:7890")

# baseprompt
prompts = BasePrompt()
prompts.build_prompt("Give me three names of cats.")

# ieprompt
in_context_examples = [{"Input": "Barcelona defeated Real Madrid 3-0 in a La Liga match on Saturday.",
                        "Output": "[{'E': 'Organization', 'W': 'Barcelona'}, {'E': 'Organization', 'W': 'Real Madrid'}, {'E': 'Competition', 'W': 'La Liga'}]"}]
ieprompts = IEPrompt(task='ner')
ieprompts.build_prompt(prompt="Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday.", examples=in_context_examples)

# cotprompt
question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
in_context_examples = [{"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                        "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10."}]

zeroshot_prompts = ZeroshotCoTPrompt()
zeroshot_prompts.build_prompt(question)

fewshot_prompts = FewshotCoTPrompt()
fewshot_prompts.build_prompt(question,
                             in_context_examples = in_context_examples,
                             n_shots = 1)

# batchprompt
batch_prompt = BatchPrompt()
batch_prompt.build_prompt([prompts, ieprompts,zeroshot_prompts, fewshot_prompts])
batch_prompt.get_openai_result(engine = "gpt-3.5-turbo-0301")
batch_prompt.parse_response()
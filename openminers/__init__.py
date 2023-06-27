# Base Miner imports
from .base.miner import BaseMiner as BaseMiner
from .base.prompting_miner import BasePromptingMiner as BasePromptingMiner
from .base.prompting_miner_hf import HuggingFaceMiner as HuggingFaceMiner
from .base.embedding_miner import BaseEmbeddingMiner as BaseEmbeddingMiner

# Miner imports.
from .text_to_text.template.miner import TemplateMiner as TemplateMiner
from .text_to_text.gpt4all.miner import GPT4ALLMiner as GPT4ALLMiner
from .text_to_text.AI21.miner import AI21Miner as AI21Miner
from .text_to_text.AlephAlpha.miner import AlephAlphaMiner as AlephAlphaMiner
from .text_to_text.cohere.miner import CohereMiner as CohereMiner
from .text_to_text.gooseai.miner import GooseMiner as GooseMiner
from .text_to_text.openai.miner import OpenAIMiner as OpenAIMiner
from .text_to_text.stabilityai.miner import StabilityAIMiner as StabilityAIMiner
from .text_to_text.cerebras.miner import CerebrasMiner as CerebrasMiner
from .text_to_text.falcon.miner import FalconMiner as FalconMiner
from .text_to_text.hermes.miner import HermesMiner as HermesMiner

# Lower case imports
from .text_to_text.template.miner import TemplateMiner as template
from .text_to_text.gpt4all.miner import GPT4ALLMiner as gpt4all
from .text_to_text.AI21.miner import AI21Miner as ai21
from .text_to_text.AlephAlpha.miner import AlephAlphaMiner as alephalpha
from .text_to_text.cohere.miner import CohereMiner as cohere
from .text_to_text.gooseai.miner import GooseMiner as goose
from .text_to_text.openai.miner import OpenAIMiner as openai
from .text_to_text.stabilityai.miner import StabilityAIMiner as stability
from .text_to_text.cerebras.miner import CerebrasMiner as cerebras
from .text_to_text.falcon.miner import FalconMiner as falcon

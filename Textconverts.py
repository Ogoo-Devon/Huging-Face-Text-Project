# TEXT_TO_STORY WITH GPT

!pip install openai==0.28 langchain langchain_community

import os
import openai
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"]

def generate_story():
  template = f"""You are a story writer, In less than 150 words, Generate a story based on a simple narrative
  CONTEXT: A child holding a flyer in a stadium filled with people that reads, LET LOVE LEAD!!!..."""
  STORY = """ """
  return template

template = generate_story()

prompt = PromptTemplate(template=template, input_variables=[])

my_story = LLMChain(
    llm=OpenAI(model_name = "gpt-3.5-turbo", temperature = 1),
    prompt=prompt)

story = my_story.predict()
print(story)





# TEXT_TO_SPEECH

!pip install --upgrade pip
!pip install --upgrade transformers sentencepiece datasets[audio]
!pip install torch

from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser("""Amid the stadium’s roar, a child clutched a flyer: 'LET LOVE LEAD'.
 As the crowd cheered, his innocent gaze scanned the sea of faces,
 holding onto a belief that kindness could turn this frenzy into unity.
 In that moment, the stadium wasn’t just filled with fans but with a shared hope for love.""", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="""Amid the stadium’s roar, a child clutched a flyer: 'LET LOVE LEAD'.
 As the crowd cheered, his innocent gaze scanned the sea of faces,
 holding onto a belief that kindness could turn this frenzy into unity.
 In that moment, the stadium wasn’t just filled with fans but with a shared hope for love.""", return_tensors="pt")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)

txt = """Amid the stadium’s roar, a child clutched a flyer: 'LET LOVE LEAD'.
 As the crowd cheered, his innocent gaze scanned the sea of faces,
 holding onto a belief that kindness could turn this frenzy into unity.
 In that moment, the stadium wasn’t just filled with fans but with a shared hope for love."""





# TEXT_TO_VIDEO

!pip install diffusers
!pip install huggingface_hub
!pip install safetensors

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda"
dtype = torch.float16

step = 4  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

output = pipe(prompt="A Nurse attending to a patient in the hospital", guidance_scale=1.0, num_inference_steps=step)
export_to_gif(output.frames[0], "animation.gif")





# TEXT_TO_IMAGE

import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU found!")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. No GPU detected.")

!pip install diffusers

from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
import torch

import tensorflow as tf
tf.keras.backend.clear_session()
#To clear some memory if CUDA is out of memory.

pipe = DiffusionPipeline.from_pretrained(
    "mann-e/Mann-E_Dreams", torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

image = pipe(
  prompt="A Health personnel with a fitted eye glasses coding on her system",
  num_inference_steps=8,
  guidance_scale=3,
  width=768,
  height=768,
  clip_skip=1
).images[0]
image.save("a_nurse.png")

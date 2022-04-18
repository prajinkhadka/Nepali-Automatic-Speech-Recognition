import os
import gradio as gr
import librosa
import torch

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
processor = Wav2Vec2Processor.from_pretrained("prajin/wav2vec2-large-xlsr-300m-nepali", use_auth_token="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")
from transformers import Wav2Vec2ProcessorWithLM
# processor_lm = Wav2Vec2ProcessorWithLM.from_pretrained("prajin/wav2vec2-large-xlsr-300m-nepali")
model = Wav2Vec2ForCTC.from_pretrained("prajin/wav2vec2-large-xlsr-300m-nepali", use_auth_token="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")
import librosa
import soundfile as sf 
import torchaudio
import numpy as np 
from loguru import logger


def process_audio_file(file):
   logger.debug("herheurhe inside process speech")
   import os 
   os.system("cp {} {}".format(file, "app"))
   # _, sampling_rate = sf.read(file)
   # logger.debug("eta samma ayo ta")
   # logger.debug(sampling_rate)
   # speech = torchaudio.load(file)
   # speech = speech[0].numpy().squeeze()
   # logger.debug(speech)
   # speech = librosa.resample(np.asarray(speech), sampling_rate, 16_000)
   # speech_arr = speech[1]

       
   logger.debug("process vitra bata", file)
   logger.debug(file)
   data, sr = librosa.load(file)
   logger.debug(data)
   logger.debug(sr)
   if sr != 16000:
      data = librosa.resample(data, sr, 16000)
      logger.debug(data)

   logger.debug(data.shape)
   print(data.shape)    

   inputs = processor(data, sampling_rate=16000, return_tensors="pt")
   logger.debug(type(inputs))
   with torch.no_grad():
      logits = model(**inputs).logits
         
   predicted_ids = torch.argmax(logits, dim=-1)
   logger.debug(predicted_ids)
   transcription = processor.batch_decode(predicted_ids)
   print(transcription)
   logger.debug(transcription)

   
   return transcription
    

def transcribe(file_mic, file_upload):
   logger.debug("That's it, beautiful and simple logging!")
   print("khoi ta")
   warn_output = ""
   if (file_mic is not None) and (file_upload is not None):
      logger.debug("That's it!")
      warn_output = "WARNING: You've uploaded an audio file and used the microphone. The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
      file = file_mic
   elif (file_mic is None) and (file_upload is None):
      logger.debug("That's!")
      return "ERROR: You have to either use the microphone or upload an audio file"
   elif file_mic is not None:
      logger.debug("That's! yehidhfjadfjdajfhdjf")

      file = file_mic
   else:
      file = file_upload

   print(type(file))
   logger.debug(type(file))
   print(file)
   logger.debug(file)

   transcription = process_audio_file(file)

   print(type(file))
   logger.debug(type(file))
   print(file)
   logger.debug(file)


   return warn_output + transcription[0]

    
iface = gr.Interface(
   fn=transcribe, 
   inputs=[
      gr.inputs.Audio(source="microphone", type='filepath', optional=True),
      gr.inputs.Audio(source="upload", type='filepath', optional=True),
   ],
   outputs="text",
   layout="horizontal",
   theme="huggingface",
   title="Automatic Nepali Speech Recognition",
   description="Interface for Nepali ASR system.",
   article = "<p style='text-align: center'><a href='https://github.com' target='_blank'>Click here for the source code, models. </a>",
   enable_queue=True,
   allow_flagging="manual"
)
iface.launch(server_port = 5000, server_name="0.0.0.0", debug=True)
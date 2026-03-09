import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.freeze()
model = model.to(device)

model.cur_decoder = "ctc"
ctc_text = model.transcribe(['sample_audio_infer_ready.wav'], batch_size=1, logprobs=False, language_id='gu')[0]
print("CTC Decoder: ", ctc_text)

model.cur_decoder = "rnnt"
rnnt_text = model.transcribe(['sample_audio_infer_ready.wav'], batch_size=1, language_id='gu')[0]
print("RNN-T Decoder: ", rnnt_text)
import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v2"
)
model.config.forced_decoder_ids = None

# load dummy dataset and read audio files
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sampling_rate = ds[0]["audio"]["sampling_rate"]
sample = (
    torch.tensor(ds[0]["audio"]["array"]).unsqueeze(0).repeat(8, 1).unbind(0)
)


input_features = processor(
    sample, sampling_rate=sampling_rate, return_tensors="pt"
).input_features

# # generate token ids
# predicted_ids = model.generate(input_features)
# # decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

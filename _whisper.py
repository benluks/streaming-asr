from whisper import load_model

from matplotlib import pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel, pipeline

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
text = "Replace me by any [MASK] you'd like."


unmasker = pipeline('fill-mask', model='distilbert-base-uncased')

output = unmasker(text)

encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)

pos_embeddings = next(model.embeddings.position_embeddings.parameters())*100
plt.imshow(pos_embeddings.T.detach().numpy())
plt.show()

# model = load_model("tiny.en")

# model.transcribe("/Users/ben/iwonder.wav")
# model

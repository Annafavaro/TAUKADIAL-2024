from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence"]

model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
embeddings = model.encode(sentences)
print(embeddings)
print(embeddings.shape)
numpy_array = embeddings.numpy()
#save(output_dir + base_name + '.npy', numpy_array)
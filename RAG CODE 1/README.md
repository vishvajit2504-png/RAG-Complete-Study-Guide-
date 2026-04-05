# RAG-Complete-Study-Guide-
This part of code will cover all the basics topics 

1) Embeddings 
2) chunking
3) Indexing
3) Storing the index
4) Retriveing index

# check_ollama_running
This Functions checks if ollama is running if not this runs ollama by running the command ollama list.
You must ollama pull the llama3.2:1b model and nomic-embed-text in your local to run this without any issues

# get_embedding
This Function call the localhost/api/embeddings post request of the model and gets the embeddings in results pretty basic stuff

# get_embeddings_batch
This functions runs the get_embedding in batch format the parts we need to focus on here is normalization we normalizse the length the of the vectors to unit lenght the reason been after normalization the dot product of the vectors is cosine similairty.
We are not covering the actually cosine similarty and types of similarity in the part that will be in the next part

# load_all_pdfs
load all the pdfs into text format and add info about each document 

# chunk_document
chunk_document into the right size of 500 and overlap 

# build_index
build index for each chunk and put it inside the get_embeddings functions and add it inot the index



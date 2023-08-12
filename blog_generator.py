from transformers import GPT2LMHeadModel, GPT2Tokenizer

#Load Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = tokenizer.eos_token_id)
 
#Tokenize Sentence
sentence = "I love chocolates."
input_ids = tokenizer.encode(sentence, return_tensor='pt')

output = model.generate(input_ids, max_length = 130, num_beams = 5, no_repeat_ngram_size= 2, early_stopping= True)

tokenizer.decode(output[0], skip_special_tokens =True)



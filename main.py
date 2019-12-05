def main():
	if len(sys.argv) != 2 or sys.argv[1] not in {"WORD","CHAR"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [WORD/CHAR]")
			exit()

	print("Running preprocessing...")
	# TODO: call preprocess function
	train, vocab = get_data('data/fls.txt','data/els.txt','data/flt.txt','data/elt.txt')
	print("Preprocessing complete.")

	model_args = (FRENCH_WINDOW_SIZE,len(french_vocab),ENGLISH_WINDOW_SIZE, len(english_vocab))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args)


	# TODO:
	# Train and test model for 1 epoch.
	train(model, train_french, train_english, eng_padding_index)
	perplexity, persymacc = test(model, test_french, test_english, eng_padding_index)
	print('This is the perplexity: {}'.format(perplexity))
	print('This is the %symacc: {}'.format(persymacc))

if __name__ == '__main__':
   main()

 # English → Yoruba Neural Machine Translation

This project explores neural machine translation from **English to Yoruba**, a low-resource language. The dataset of Menyo20k and Mafand(eng-yor) contains too few aligned examples to reliably train a custom translation model from scratch. With limited data, a model with substantial capacity can memorize the training examples instead of learning general translation patterns which we show in our notebooks.

## Approach

Rather than building a model from zero, we fine-tuned Meta’s **NLLB (No Language Left Behind)** model. NLLB already contains abstract multilingual representations learned from high-resource languages. These representations provide a strong starting point for Yoruba, allowing the model to learn useful English–Yoruba mappings from a relatively small dataset and generalize better than a custom-built model.

The workflow is:

1. Prepare and align the English–Yoruba sentence pairs.
2. Tokenize the data with the NLLB tokenizer and configure the English and Yoruba language codes.
3. Fine-tune the pretrained NLLB model on the translation pairs.
4. Evaluate progress with BLEU and inspect generated translations.
5. Serve the trained model through an HTML interface for interactive testing.

## Results and current status

The BLEU score improved from approximately **14 to nearly 20**, indicating that the model is capturing the gist of the source sentences and producing increasingly useful translations. Training is not finished yet: the project reached the compute limits of the Google Colab environment, including its usage restrictions, before the model could be fully optimized.

We plan to return to the project and continue training and evaluation when more compute is available. Further data cleaning, additional aligned examples, longer training, and broader validation may improve quality.

## Demo

The model has also been deployed to an HTML page for interactive inference. Feel free to deploy and test it if your laptop or cloud environment has sufficient memory and compute for the model.

## Limitations

- Yoruba is underrepresented in available training data compared with high-resource languages.
- The small dataset increases the risk of memorization and limits evaluation reliability.
- BLEU is only one measure of translation quality and does not fully capture fluency, meaning, or cultural context.
- Running and fine-tuning NLLB may require more memory and compute than a standard local environment provides.

This is an ongoing effort toward a more capable English–Yoruba translation system.



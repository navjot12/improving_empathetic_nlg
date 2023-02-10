# Work done so far:

### [MoEL](https://github.com/navjot12/improving_empathetic_nlg/tree/main/MoEL)
1. Cloned [MoEL](https://github.com/HLTCHKUST/MoEL) and attempted retraining the model. Raised [Issue#12](https://github.com/navjot12/improving_empathetic_nlg/issues/12).
1. Integrated wandb for logging. See [dashboard](https://wandb.ai/improving-empathetic-nlg/moel) for history of runs and logs.
1. Modifications made to MoEL:
	<ol> 
		<li> Decreased check_iter to 1500 from 2000 to increase frequency of logging. </li>
		<li> Increased patience to 3 from 2. Patience is increased every time validation data evaluation does not show improvement in performance (which is done after every check_iter number of iterations). This change makes the model train for another check_iter number of training batches before deciding that the model is overfitting (and stop training). Experiments have shown that the model keeps overfitting if patience is allowed to increases to 20. </li>
		<li> Started reporting actual calculated perplexity instead of exponential loss. </li>
		<li> Bug fix: Validation accuracy over-wrote train accuracy every time validation data was evaluated. </li>
	</ol>
1. To see if MoEL does better if number of distinct emotions are reduced, 32 emotions were grouped into 16 according to the categorization [here](https://docs.google.com/spreadsheets/d/1lBUdjVdTJ17kOqA6RRwNHglp0MKJumMNQEkeP_Xu720/edit?usp=sharing). This is available in the [16_emotions](https://github.com/navjot12/improving_empathetic_nlg/tree/16_emotions) branch.</li>

### [PEC](https://github.com/navjot12/improving_empathetic_nlg/tree/main/pec)
1. [PEC](https://huggingface.co/datasets/pec) contains 281,163 instances in total which was filtered according to following criterion (applied in order):
	<ol>
		<li> Context only has 1 speaker. </li>
		<li> The context sentences have at least 20 words in total. </li>
		<li> Response speakers' personas have at least 25 sentences. </li>
		<li> Response utterance has at least 10 words. </li>
	</ol>
	</li>
1. [Longformer](https://huggingface.co/docs/transformers/model_doc/longformer) was used to derive persona embeddings for persona sentences after filtering PEC.
	<ol>
		<li> Persona embedding is a 3,072 dimension vector, calculated by averaging the concatenation of last 4 layers of longformer, across all word in a persona. </li>
		<li> Local attention was configured for every persona sentence, while global attention was configured across separator tokens </li>
	</ol>
1. PEC was transformed from to the format required by MoEL: parallel vectors.


### [Emotion Classifier](https://github.com/navjot12/improving_empathetic_nlg/tree/main/ed_classifier)
1. MoEL requires emotion annotation along with context. To augment PEC with it, a BERT-based emotion classifier was built.

# Command to launch MoEL training:

`python3 main.py --model experts  --label_smoothing --noam --emb_dim 300 --hidden_dim 300 --hop 1 --heads 2 --topk 5 --cuda --pretrain_emb --softmax --basic_learner --schedule 10000`

Optional arguments:
1. `--wandb_project moel` : To log the training run's parameters on [wandb](https://wandb.ai/improving-empathetic-nlg/moel).
1. `--ed_16` : To train MoEL on the empathetic-dialogue dataset with 16 emotions. This needs the numpy arrays required by MoEL to be present in ed_16 folder.
1. `--pec_2` : To train MoEL on pec dataset with 2 emotions. This needs the numpy arrays required by MoEL to be present in pec_2 folder.
1. `--pec_32` : To train MoEL on pec dataset with 32 emotions. This needs the numpy arrays required by MoEL to be present in pec_32 folder.
1. `--use_persona` : To train MoEL with persona embeddings pec dataset with 32 emotions. This needs the corresponding numpy arrays to be present.

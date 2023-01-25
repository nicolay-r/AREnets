## Input samples format

Input data might be in a form of `.jsoln`
[[example]](../tutorials/_data/sample-train.jsonl)
or `csv`
[[example]](../tutorials/_data/sample-train.tsv.gz).

Input represents list/rows of *samples*. 
Every sample contains text and mentioned **opinion** in it, i.e. `source->target` relation.

Sample contain the following mandatory parameters:
* `id` (type:`uint`) -- sample identifier
* `label` (type: `int`) -- for training only, denotes a class; 
    * value in range `[0, c]`, where `c` corresponds to classes amount.
* `text` (type: `str` or `list`) -- string of terms, separated by ` ` (whitespace), or list of terms in case of `jsonl` fomat;

Optional parameters:  
* `s_ind` (type: `int`) -- index of the **source** term in `text` string/list;
* `t_ind` (type: `int`) -- index of the **target** term in `text` string/list;
* `opinion_id` (type: `uint`) -- used for grouping opinions, denoting index in group;
* `entities` (type: `str`) -- comma separated term indices which corresponds to entities, in order of their appearance in text
* `frames` (type: `str`) -- comma separated term indices which corresponds to connotation frames, in order of their appearance in text; 
  important for sentiment-classification related tasks;
* `frame_connots_uint` (type: `str`) -- comma separated scores of the in set of three scale `int` values `{-1, 0, 1}`;
* `syn_subjs` (type: `str`) -- comma separated indices, synonymous to source `s_ind`;
* `syn_objs` (type: `str`) -- comma separated indices, synonymous to target `t_ind`;
* `pos_tags` (type: `str`) -- comma separated part-of-speech tags, with length exact the same as terms count of `text`;

## Embedding details

We support `model.txt` format, which provides:
* first row -- shape of the embedding matrix
* word and its vector in every row

Embeddings could be obtained from [NLPL repository](http://vectors.nlpl.eu/repository/)
* See how we do this in the following [downloading script](../tutorials/_data/download_embedding.sh)

Text-based embeddings will be then converted into `vocab.txt` and `embedding.npz` matrix.
[[see code implementation]](../arenets/emb_converter.py)
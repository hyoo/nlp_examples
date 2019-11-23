import pandas as pd
from nlu_seq2seq_load_data import load_atis

t2i_train, s2i_train, in2i_train, i2t_train, i2s_train, i2in_train, \
    input_tensor_train, target_tensor_train, \
    query_data_train, intent_data_train, intent_data_label_train, slot_data_train = load_atis('ms_cntk_atis.train.pkl')

pd.set_option('display.max_colwidth', -1)
df = pd.DataFrame({'query': query_data_train, 'intent': intent_data_train,
                   'slot filling': slot_data_train})
df_small = pd.DataFrame(columns=['query', 'intent', 'slot filling'])
j = 0
for i in df.intent.unique():
    df_small.loc[j] = df[df.intent == i].iloc[0]
    j = j + 1

# df_small

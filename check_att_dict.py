from dfa.utils import unpickle_binary

if __name__ == '__main__':

    att_dict = unpickle_binary('/Users/cschaefe/att_score_dict.pkl')
    scores = [(item_id, val[1]) for item_id, val in att_dict.items()]
    scores.sort(key=lambda x: x[1])

    print(scores[:10])
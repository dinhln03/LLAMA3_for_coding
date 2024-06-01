
from transformers import RobertaConfig

from modeling.hf_head.modeling_roberta_parsing import RobertaForGraphPrediction
from modeling.sequence_labeling import SequenceLabeling

if __name__ == '__main__':
    config = RobertaConfig(graph_head_hidden_size_mlp_arc=100, graph_head_hidden_size_mlp_rel=100, dropout_classifier=0.1)
    #config.graph_head_hidden_size_mlp_arc = 100
    model = RobertaForGraphPrediction(config)

    SequenceLabeling(

    )

    # 1. GIVE IT TO PYTORCH LIGHTNING
    # 2. DEFINE DATA MODULE FOR PARSING --> INPUT + LOSS: TRY TO FIT
    # 3. Prediction (recover full graph after bpes-
    breakpoint()

from utils import test_eval
from model import *


class ARCDetector:
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.model = ARC(**model_config).to(train_config['device'])

    def train(self):
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['weight_decay'])

        for e in range(self.train_config['epochs']):
            for didx, train_data in enumerate(self.data['train']):
                self.model.train()
                train_graph = self.data['train'][didx].graph.to(self.train_config['device'])
                residual_embed = self.model(train_graph)
                loss = self.model.cross_attn.get_train_loss(residual_embed, train_graph.ano_labels,
                                                            self.model_config['num_prompt'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print('Finish Training for {} epochs!'.format(self.train_config['epochs']))
        # Evaluation
        test_score_list = {}
        self.model.eval()
        for didx, test_data in enumerate(self.data['test']):
            test_graph = test_data.graph.to(self.train_config['device'])
            labels = test_graph.ano_labels
            shot_mask = test_graph.shot_mask.bool()

            query_labels = labels[~shot_mask].to(self.train_config['device'])
            residual_embed = self.model(test_graph)

            query_scores = self.model.cross_attn.get_test_score(residual_embed, test_graph.shot_mask,
                                                                  test_graph.ano_labels)
            test_score = test_eval(query_labels, query_scores)
            # Store the test scores in the dictionary
            test_data_name = self.train_config['testdsets'][didx]
            test_score_list[test_data_name] = {
                'AUROC': test_score['AUROC'],
                'AUPRC': test_score['AUPRC'],
            }
        return test_score_list

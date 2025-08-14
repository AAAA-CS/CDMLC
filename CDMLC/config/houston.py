from collections import OrderedDict

config = OrderedDict()
config['data_path'] = 'E:/liuwj/datasets'
config['save_path'] = '/results'
config['source_data'] = 'Chikusei_imdb_128.pickle'
config['target_data'] = 'Houston/Houston.mat'
config['target_data_gt'] = 'Houston/Houston_gt.mat'
config['gpu'] = 0

config['src_label_mapping'] = {0: 'water', 1: 'baress', 2: 'baresp', 3: 'baresf', 4: 'naturalp', 5: 'weedsmf',
                               6: 'forest', 7: 'grass', 8: 'ricefg', 9: 'riceffstage', 10: 'rowc', 11: 'plastich',
                               12: 'mannan', 13: 'mandark', 14: 'manblue', 15: 'manred', 16: 'mangrass', 17: 'asphalt'}

config['tar_label_mapping'] = {0: 'healthygrass', 1: 'stressedgrass', 2: 'syntheticgrass', 3: 'trees', 4: 'soil', 5: 'water',
                               6: 'residential', 7: 'commercial', 8: 'road', 9: 'highway', 10: 'railway', 11: 'parkingLot1',
                               12: 'parkingLot2', 13: 'tenniscourt', 14: 'runningtrack'}

config['queue_label_mapping_src'] = {0: 'water', 1: 'baress', 2: 'baresp', 3: 'baresf', 4: 'naturalp', 5: 'weedsmf',
                                     6: 'forest', 7: 'grass', 8: 'ricefg', 9: 'riceffstage', 10: 'rowc', 11: 'plastich',
                                     12: 'mannan', 13: 'mandark', 14: 'manblue', 15: 'manred', 16: 'mangrass', 17: 'asphalt'}

config['queue_label_mapping_tar'] = {0: 'healthygrass', 1: 'stressedgrass', 2: 'syntheticgrass', 3: 'trees', 4: 'soil', 5: 'water',
                                     6: 'residential', 7: 'commercial', 8: 'road', 9: 'highway', 10: 'railway', 11: 'parkingLot1',
                                     12: 'parkingLot2', 13: 'tenniscourt', 14: 'runningtrack'}

train_opt = OrderedDict ()
train_opt['patch_size'] = 9
train_opt['batch_task'] = 1
train_opt['num_ways'] = 5
train_opt['num_shots'] = 1
train_opt['episode'] = 10000
train_opt['lr'] = 1e-2
train_opt['weight_decay'] = 1e-4
train_opt['dropout'] = 0.1

train_opt['d_emb'] = 128
train_opt['src_input_dim'] = 128
train_opt['tar_input_dim'] = 144
train_opt['n_dim'] = 100
train_opt['class_num'] = 15
train_opt['shot_num_per_class'] = 1
train_opt['query_num_per_class'] = 19

train_opt['test_class_num'] = 15
train_opt['test_lsample_num_per_class'] = 1

train_opt['dic_num_classes_src'] = 18
train_opt['dic_num_classes_tar'] = 15
train_opt['dic_len'] = 100
train_opt['move_momentum'] = 0.9
train_opt['sample_size'] = 1000

train_opt['n_clusters'] = 15

config['train_config'] = train_opt
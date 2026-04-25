from collections import OrderedDict

config = OrderedDict()
config['data_path'] =  'F:/liuwj/datasets'
config['source_data'] = 'Chikusei_imdb_128.pickle'
config['target_data'] = 'IP/indian_pines_corrected.mat'
config['target_data_gt'] = 'IP/indian_pines_gt.mat'
config['test_lsample_num_per_class'] = 5

train_opt = OrderedDict ()

# 标签映射
train_opt['src_label_mapping'] = {0: 'water', 1: 'baress', 2: 'baresp', 3: 'baresf', 4: 'naturalp', 5: 'weedsmf',
                               6: 'forest', 7: 'grass', 8: 'ricefg', 9: 'riceffstage', 10: 'rowc', 11: 'plastich',
                               12: 'mannan', 13: 'mandark', 14: 'manblue', 15: 'manred', 16: 'mangrass', 17: 'asphalt'}

train_opt['tar_label_mapping'] = {0: 'alfalfa', 1: 'cornno', 2: 'cornni', 3: 'corn', 4: 'grassp', 5: 'garsstree',
                               6: 'grasspm', 7: 'hatwin', 8: 'oats', 9: 'soyno', 10: 'soymi', 11: 'soyclean',
                               12: 'wheat', 13: 'woods', 14: 'BGTD', 15: 'SST'}

train_opt['queue_label_mapping_src'] = {0: 'water', 1: 'baress', 2: 'baresp', 3: 'baresf', 4: 'naturalp', 5: 'weedsmf',
                                     6: 'forest', 7: 'grass', 8: 'ricefg', 9: 'riceffstage', 10: 'rowc', 11: 'plastich',
                                     12: 'mannan', 13: 'mandark', 14: 'manblue', 15: 'manred', 16: 'mangrass', 17: 'asphalt'}

train_opt['queue_label_mapping_tar'] = {0: 'alfalfa', 1: 'cornno', 2: 'cornni', 3: 'corn', 4: 'grassp', 5: 'garsstree',
                                     6: 'grasspm', 7: 'hatwin', 8: 'oats', 9: 'soyno', 10: 'soymi', 11: 'soyclean',
                                     12: 'wheat', 13: 'woods', 14: 'BGTD', 15: 'SST'}

# 特征维度
train_opt['patch_size'] = 9
train_opt['src_input_dim'] = 128
train_opt['tar_input_dim'] = 200
train_opt['n_dim'] = 100
train_opt['d_emb'] = 64
# 元任务设置
train_opt['episode'] = 10000
train_opt['class_num'] = 16
train_opt['shot_num_per_class'] = 1
train_opt['query_num_per_class'] = 19
# 类别数量
train_opt['test_class_num'] = 16
train_opt['dic_num_classes_src'] = 18
train_opt['dic_num_classes_tar'] = 16
train_opt['n_clusters'] = 16
# 训练参数
train_opt['lr'] = 1e-2
train_opt['dic_len'] = 100
train_opt['move_momentum'] = 0.9
train_opt['sample_size'] = 1000

train_opt['lambda'] = 1
train_opt['Beta'] = 0.01

config['train_config'] = train_opt
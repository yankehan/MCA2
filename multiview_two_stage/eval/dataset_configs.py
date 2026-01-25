DATASET_CONFIGS = {
    'olid': {
        'type': 'contrastive-friendly',
        'description': '攻击性语言检测 - 偏视图互补',
        'num_epochs': 20,
        'learning_rate': 0.01,
        'batch_size': 256,
        'lambda_recon':0.05,
        'lambda_contrastive': 0.95,
        'score_weight_recon': 0.03,
        'score_weight_consistency': 0.95,
    },
    'hate_speech': {
        'type': 'contrastive-friendly',
        'description': '仇恨言论检测 - 偏视图互补',
        'num_epochs': 10,
        'learning_rate': 0.01,
        'batch_size': 256,
        'lambda_recon': 1,
        'lambda_contrastive': 1,
        'score_weight_recon': 0.3,
        'score_weight_consistency': 0.4,
    },
    'covid_fake': {
        'type': 'reconstruction-friendly',
        'description': '新冠假新闻检测 - 偏重构',
        'num_epochs': 80,
        'learning_rate': 0.002,
        'batch_size': None,
        'lambda_recon': 5.0,
        'lambda_contrastive': 0.1,
        'score_weight_recon': 0.95,
        'score_weight_consistency': 0.03,
    },
    'liar2': {
        'type': 'reconstruction-friendly',
        'description': '谎言检测 - 偏重构',
        'num_epochs': 100,
        'learning_rate': 0.001,
        'batch_size': None,
        'lambda_recon': 5.0,
        'lambda_contrastive': 0.1,
        'score_weight_recon': 5,
        'score_weight_consistency': 0.03,
    },
    'email_spam': {
        'type': 'reconstruction-friendly',
        'description': '邮件垃圾检测 - 偏重构',
        'num_epochs': 45,
        'learning_rate': 0.001,
        'batch_size': 256,
        'lambda_recon': 5,
        'lambda_contrastive': 0.1,
        'score_weight_recon': 5,
        'score_weight_consistency': 0.01,
    },
    'smsspam': {
        'type': 'reconstruction-friendly',
        'description': '短信垃圾检测 - 偏重构',
        'num_epochs': 30,
        'learning_rate': 0.001,
        'batch_size': 256,
        'lambda_recon': 1,
        'lambda_contrastive': 1,
        'score_weight_recon': 0.3,
        'score_weight_consistency': 0.4,
    },
    'bbc': {
        'type': 'reconstruction-friendly',
        'description': 'BBC新闻分类 - 偏重构',
        'num_epochs': 100,
        'learning_rate': 0.001,
        'batch_size': None,
        'lambda_recon': 5,
        'lambda_contrastive': 0.1,
        'score_weight_recon': 0.95,
        'score_weight_consistency': 0.03,
    },
    'agnews': {
        'type': 'reconstruction-friendly',
        'description': 'AGNews新闻分类 - 偏重构',
        'num_epochs': 1,
        'learning_rate': 0.001,
        'batch_size': 256,
        'lambda_recon': 5.0,
        'lambda_contrastive': 0.1,
        'score_weight_recon': 0.95,
        'score_weight_consistency': 0.03,
    },
    'movie_review': {
        'type': 'reconstruction-friendly',
        'description': '偏对比学习',
        'num_epochs': 30,
        'learning_rate': 0.001,
        'batch_size': 256,
        'lambda_recon': 1,
        'lambda_contrastive': 1,
        'score_weight_recon': 0.3,
        'score_weight_consistency': 0.4,
    },
    'N24News': {
        'type': 'reconstruction-friendly',
        'description': '偏对比学习',
        'num_epochs': 1,
        'learning_rate': 0.001,
        'batch_size': 256,
        'lambda_recon': 1,
        'lambda_contrastive': 1,
        'score_weight_recon': 0.3,
        'score_weight_consistency': 0.4,
    },
}
def get_dataset_config(dataset_name):
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name].copy()
    else:
        return {
            'type': 'balanced',
            'description': '未知数据集 - 使用平衡配置',
            'num_epochs': 200,
            'learning_rate': 0.002,
            'batch_size': None,
            'lambda_recon': 1.0,
            'lambda_contrastive': 1.0,
            'score_weight_recon': 0.3,
            'score_weight_consistency': 0.4,
        }
def print_dataset_config(dataset_name):
    config = get_dataset_config(dataset_name)
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"类型: {config['type']}")
    print(f"说明: {config['description']}")
    print(f"{'='*60}")
    print(f"训练参数:")
    print(f"  - num_epochs: {config['num_epochs']}")
    print(f"  - learning_rate: {config['learning_rate']}")
    batch_size = config.get('batch_size', None)
    if batch_size is None:
        print(f"  - batch_size: None (全批次训练)")
    else:
        print(f"  - batch_size: {batch_size}")
    print(f"损失权重:")
    print(f"  - lambda_recon: {config['lambda_recon']}")
    print(f"  - lambda_contrastive: {config['lambda_contrastive']}")
    print(f"分数权重:")
    print(f"  - score_weight_recon: {config['score_weight_recon']}")
    print(f"  - score_weight_consistency: {config['score_weight_consistency']}")
    print(f"{'='*60}\n")
    return config
def get_all_datasets_by_type():
    contrastive_friendly = []
    reconstruction_friendly = []
    balanced = []
    for dataset, config in DATASET_CONFIGS.items():
        if config['type'] == 'contrastive-friendly':
            contrastive_friendly.append(dataset)
        elif config['type'] == 'reconstruction-friendly':
            reconstruction_friendly.append(dataset)
        else:
            balanced.append(dataset)
    return {
        'contrastive-friendly': contrastive_friendly,
        'reconstruction-friendly': reconstruction_friendly,
        'balanced': balanced,
    }
if __name__ == '__main__':
    print("="*80)
    print("数据集配置总览")
    print("="*80)
    datasets_by_type = get_all_datasets_by_type()
    print("\n【类型A: 对比学习友好型】")
    print("适合使用: 重构 + 对比学习")
    for dataset in datasets_by_type['contrastive-friendly']:
        print(f"  - {dataset}")
    print("\n【类型B: 重构友好型】")
    print("适合使用: 纯重构或重构主导")
    for dataset in datasets_by_type['reconstruction-friendly']:
        print(f"  - {dataset}")
    print("\n【类型C: 平衡型】")
    print("适合使用: 平衡配置")
    for dataset in datasets_by_type['balanced']:
        print(f"  - {dataset}")
    print("\n" + "="*80)
    print("详细配置:")
    print("="*80)
    for dataset in DATASET_CONFIGS.keys():
        print_dataset_config(dataset)

import csv
import random
from pathlib import Path

random.seed(0)

ROOT = Path('eval/results')
MGTBENCH_PATTERN = 'mgtbench/mgtbench_*_GPT2.csv'
CHEAT_FILE = ROOT / 'cheat/cheat_GPT2.csv'
HC3_FILE = ROOT / 'hc3/hc3_GPT2.csv'


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = [r for r in reader]
    return cols, rows


def load_data_by_group():
    """Load data from all groups and return a dict of group_name: (columns, rows)"""
    data = {}
    
    # Load mgtbench files
    mgtbench_files = sorted(ROOT.glob(MGTBENCH_PATTERN))
    if not mgtbench_files:
        raise SystemExit('No mgtbench files found')
    
    mgtbench_rows = []
    mgtbench_cols = set()
    for path in mgtbench_files:
        cols, rows = read_rows(path)
        mgtbench_cols.update(cols)
        mgtbench_rows.extend(rows)
    random.shuffle(mgtbench_rows)
    data['mgtbench'] = (list(mgtbench_cols), mgtbench_rows)
    
    # Load cheat
    if not CHEAT_FILE.exists():
        raise SystemExit(f'Missing {CHEAT_FILE}')
    cheat_cols, cheat_rows = read_rows(CHEAT_FILE)
    random.shuffle(cheat_rows)
    data['cheat'] = (cheat_cols, cheat_rows)
    
    # Load hc3
    if not HC3_FILE.exists():
        raise SystemExit(f'Missing {HC3_FILE}')
    hc3_cols, hc3_rows = read_rows(HC3_FILE)
    random.shuffle(hc3_rows)
    data['hc3'] = (hc3_cols, hc3_rows)
    
    return data


def combine_groups(groups_data, group_names):
    """Combine data from specified groups into a single list of rows and unified columns"""
    all_columns = []
    combined_rows = []
    
    for group_name in group_names:
        if group_name not in groups_data:
            raise SystemExit(f'Unknown group: {group_name}')
        cols, rows = groups_data[group_name]
        # Add new columns
        for col in cols:
            if col not in all_columns:
                all_columns.append(col)
        combined_rows.extend(rows)
    
    return all_columns, combined_rows


def preprocess_and_save(rows, columns, output_path, feature_columns):
    """Preprocess rows: filter errors, clean labels, select features, and save to CSV"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_cols = [c for c in columns if c != 'error']
    if feature_columns is None:
        feature_columns = [c for c in all_cols if c not in ('gltr_model', 'source', 'label', 'ai_model')]
    else:
        missing = [c for c in feature_columns if c not in all_cols]
        if missing:
            raise SystemExit(f'Unknown feature columns: {missing}')
    
    fieldnames = feature_columns + ['target']
    cleaned_rows = []
    for row in rows:
        if row.get('error'):
            continue
        label = (row.get('label') or '').strip().lower()
        if label not in ('human', 'ai'):
            continue
        cleaned_row = {k: row.get(k, '') for k in feature_columns}
        cleaned_row['target'] = 1 if label == 'ai' else 0
        cleaned_rows.append(cleaned_row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f'Wrote {output_path} with {len(cleaned_rows)} rows')
    return len(cleaned_rows)


def create_dataset_split(train_groups, test_groups, scenario_name, feature_columns=None):
    """Create train/test split based on groups and save preprocessed datasets"""
    data = load_data_by_group()
    
    # Combine train data
    train_cols, train_rows = combine_groups(data, train_groups)
    train_path = Path('eval/train-dataset') / f'{scenario_name}_train.csv'
    train_count = preprocess_and_save(train_rows, train_cols, train_path, feature_columns)
    
    # Combine test data
    test_cols, test_rows = combine_groups(data, test_groups)
    test_path = Path('eval/train-dataset') / f'{scenario_name}_test.csv'
    test_count = preprocess_and_save(test_rows, test_cols, test_path, feature_columns)
    
    print(f'Scenario {scenario_name}: Train {train_count} rows, Test {test_count} rows')


def combine_results():
    """Original combine function - kept for compatibility"""
    data = load_data_by_group()
    all_columns, all_rows = combine_groups(data, ['mgtbench', 'cheat', 'hc3'])
    
    out = ROOT / 'mixed.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    
    print('Wrote', out, 'with', len(all_rows), 'rows')


def preprocess_data(mixed_path: Path = ROOT / 'mixed.csv',
                    output_path: Path = Path('eval/train-dataset/train.csv'),
                    feature_columns: list[str] | None = None):
    """Original preprocess function - kept for compatibility"""
    if not mixed_path.exists():
        raise SystemExit(f'Mixed file not found: {mixed_path}')
    
    with open(mixed_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f'No header found in {mixed_path}')
        
        all_cols = [c for c in reader.fieldnames if c != 'error']
        if feature_columns is None:
            feature_columns = [c for c in all_cols if c not in ('gltr_model', 'source', 'label', 'ai_model')]
        else:
            missing = [c for c in feature_columns if c not in all_cols]
            if missing:
                raise SystemExit(f'Unknown feature columns: {missing}')
        
        fieldnames = feature_columns + ['target']
        cleaned_rows = []
        for row in reader:
            if row.get('error'):
                continue
            label = (row.get('label') or '').strip().lower()
            if label not in ('human', 'ai'):
                continue
            cleaned_row = {k: row.get(k, '') for k in feature_columns}
            cleaned_row['target'] = 1 if label == 'ai' else 0
            cleaned_rows.append(cleaned_row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)
    
    print(f'Wrote {output_path} with {len(cleaned_rows)} rows')


if __name__ == '__main__':
    columns_eight_features = [
        'n_tokens',
        'frac_top10',
        'frac_top100',
        'frac_top1000',
        'frac_rest',
        'mean_rank',
        'mean_prob',
        'perplexity',
    ]
    columns_four_features = [
        'frac_top10',
        'frac_rest',
        'mean_rank',
        'perplexity',
    ]
    
    # Scenario 1: Train on hc3 + cheat, Test on mgtbench
    create_dataset_split(['hc3', 'cheat'], ['mgtbench'], 'scenario1', columns_eight_features)
    
    # Scenario 2: Train on hc3 + mgtbench, Test on cheat
    create_dataset_split(['hc3', 'mgtbench'], ['cheat'], 'scenario2', columns_eight_features)
    
    # Scenario 3: Train on mgtbench + cheat, Test on hc3
    create_dataset_split(['mgtbench', 'cheat'], ['hc3'], 'scenario3', columns_eight_features)
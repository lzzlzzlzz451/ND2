import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
用 ND2 的 config/synthetic.yaml 中的真实方程预训练 TransformerPolicy（方案三：只用真实数据）
直接硬编码前缀表达式，避免解析问题。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
 
from ND2.GDExpr import GDExpr, GDExprClass
from ND2.utils import seed_all
from new_DSO.policy import TransformerPolicy
from new_DSO.vocabulary import Vocabulary
from new_DSO.config import get_default_config
 
# ND2 和 new_DSO 之间的 token 名映射
ND2_TO_NEWDSO_TOKEN = {
    'term': 'targ',
}
 
 
def load_equations_hardcoded():
    """
    直接硬编码所有合成方程的前缀表达式。
    来源：config/synthetic.yaml 中的 GD_expr 字段。
    
    每个方程是一个 (prefix: List[str], root_type: str, name: str) 元组。
    数值系数用 '<C>' 占位。
    """
    equations = []
 
    # ── KUR ──
    # omega = omega0 + 1.0*aggr(sin(sour(x,G,A)-targ(x,G,A)), G, A)
    # → add omega0 mul <C> aggr sin sub sour x targ x
    equations.append((
        ['add', 'v1', 'mul', '<C>', 'aggr', 'sin', 'sub', 'sour', 'v2', 'targ', 'v2'],
        'node', 'KUR/omega'
    ))
 
    # ── FHN ──
    # dx = x - x**3 - y - 1*aggr(sour(x,G,A)-targ(x,G,A),G,A) / aggr(1,G,A)
    # → sub sub sub mul v2 pow3 v2 v1 div mul <C> aggr sub sour v2 targ v2 aggr <C>
    equations.append((
        ['sub', 'sub', 'sub', 'mul', 'v2', 'pow3', 'v2', 'v1',
         'div', 'mul', '<C>', 'aggr', 'sub', 'sour', 'v2', 'targ', 'v2',
         'aggr', '<C>'],
        'node', 'FHN/dx'
    ))
    # dy = 0.28 + 0.5*x - 0.04*y
    # → sub add <C> mul <C> v2 mul <C> v1
    equations.append((
        ['sub', 'add', '<C>', 'mul', '<C>', 'v2', 'mul', '<C>', 'v1'],
        'node', 'FHN/dy'
    ))
 
    # ── GR ──
    # dx = 0.2 - 0.9*x + 2.0*aggr(sour(regular(x/1.5, 2), G, A), G, A)
    # → add sub <C> mul <C> v2 mul <C> aggr sour regular div v2 <C> <C>
    equations.append((
        ['add', 'sub', '<C>', 'mul', '<C>', 'v2',
         'mul', '<C>', 'aggr', 'sour', 'regular', 'div', 'v2', '<C>', '<C>'],
        'node', 'GR/dx'
    ))
 
    # ── HCR ──
    # dx = -y - z + 0.5*aggr(sin(sour(x,G,A)-targ(x,G,A)), G, A)
    # → add sub neg v1 neg v3 mul <C> aggr sin sub sour v2 targ v2
    equations.append((
        ['add', 'sub', 'neg', 'v1', 'neg', 'v3',
         'mul', '<C>', 'aggr', 'sin', 'sub', 'sour', 'v2', 'targ', 'v2'],
        'node', 'HCR/dx'
    ))
    # dy = x + 0.165*y
    # → add v2 mul <C> v1
    equations.append((
        ['add', 'v2', 'mul', '<C>', 'v1'],
        'node', 'HCR/dy'
    ))
    # dz = 2.0 + z*(x - 5.5)
    # → add <C> mul v3 sub v2 <C>
    equations.append((
        ['add', '<C>', 'mul', 'v3', 'sub', 'v2', '<C>'],
        'node', 'HCR/dz'
    ))
 
    # ── CR ──
    # dx = -omega*y - z + 0.5*aggr(sin(sour(x,G,A)-targ(x,G,A)), G, A)
    # → add sub mul neg v1 neg v3 mul <C> aggr sin sub sour v2 targ v2
    # 注: CR 和 HCR 的 dx 类似，但 CR 用 omega 而非 y
    # -omega * y - z → sub mul neg v1 v1 v3 （简化：omega 视为 v1）
    equations.append((
        ['add', 'sub', 'mul', 'neg', 'v1', 'v1', 'v3',
         'mul', '<C>', 'aggr', 'sin', 'sub', 'sour', 'v2', 'targ', 'v2'],
        'node', 'CR/dx'
    ))
    # dy = omega*x + 0.165*y
    equations.append((
        ['add', 'mul', 'v1', 'v2', 'mul', '<C>', 'v1'],
        'node', 'CR/dy'
    ))
    # dz = 2.0 + z*(x - 5.5)
    equations.append((
        ['add', '<C>', 'mul', 'v3', 'sub', 'v2', '<C>'],
        'node', 'CR/dz'
    ))
 
    # ── LV ──
    # dx = x*(alpha - theta*x) - aggr(targ(x,G,A)*sour(x,G,A), G, A)
    # → sub mul v2 sub v1 mul v3 v2 aggr mul targ v2 sour v2
    equations.append((
        ['sub', 'mul', 'v2', 'sub', 'v1', 'mul', 'v3', 'v2',
         'aggr', 'mul', 'targ', 'v2', 'sour', 'v2'],
        'node', 'LV/dx'
    ))
 
    # ── MM ──
    # dx = -x + aggr(sour(regular(x, 2), G, A), G, A)
    # → add neg v2 aggr sour regular v2 <C>
    equations.append((
        ['add', 'neg', 'v2', 'aggr', 'sour', 'regular', 'v2', '<C>'],
        'node', 'MM/dx'
    ))
 
    # ── WC ──
    # dx = -x + aggr(sour(sigmoid(5.1*(x-1.0)), G, A), G, A)
    # → add neg v2 aggr sour sigmoid mul <C> sub v2 <C>
    equations.append((
        ['add', 'neg', 'v2', 'aggr', 'sour', 'sigmoid', 'mul', '<C>', 'sub', 'v2', '<C>'],
        'node', 'WC/dx'
    ))
 
    # ── MP ──
    # dx = x*(alpha - theta*x) + aggr(targ(x,G,A)*sour(regular(x,2),G,A), G, A)
    # → add mul v2 sub v1 mul v3 v2 aggr mul targ v2 sour regular v2 <C>
    equations.append((
        ['add', 'mul', 'v2', 'sub', 'v1', 'mul', 'v3', 'v2',
         'aggr', 'mul', 'targ', 'v2', 'sour', 'regular', 'v2', '<C>'],
        'node', 'MP/dx'
    ))
 
    # ── SIS ──
    # dx = -delta*x + aggr((1 - targ(x,G,A))*sour(x,G,A), G, A)
    # → add mul neg v3 v2 aggr mul sub <C> targ v2 sour v2
    equations.append((
        ['add', 'mul', 'neg', 'v3', 'v2',
         'aggr', 'mul', 'sub', '<C>', 'targ', 'v2', 'sour', 'v2'],
        'node', 'SIS/dx'
    ))
 
    return equations
 
 
def build_training_pairs_from_equation(prefix, root_type, vocab, max_length=30):
    """
    用 GDExpr.decompose 逐步拆解，生成训练样本。
    """
    pairs = []
    working_prefix = deepcopy(prefix)
 
    # 把 <C> 之外的数值系数也替换为 <C>
    for i, item in enumerate(working_prefix):
        if isinstance(item, str) and item not in GDExpr.word2id and item != '<C>':
            try:
                float(item)
                working_prefix[i] = '<C>'
            except (ValueError, TypeError):
                pass
 
    # 逐步 decompose
    for _ in range(len(working_prefix) + 5):
        try:
            working_prefix, policy, index = GDExpr.decompose(working_prefix, root_type)
        except Exception:
            break
 
        mapped_policy = ND2_TO_NEWDSO_TOKEN.get(policy, policy)
 
        # 构造 input: [SOS, root_type, *prefix_tokens]
        input_tokens = ['sos', root_type] + working_prefix
        input_ids = []
        for tok in input_tokens:
            mapped_tok = ND2_TO_NEWDSO_TOKEN.get(tok, tok)
            if mapped_tok in vocab.word2id:
                input_ids.append(vocab.word2id[mapped_tok])
 
        # target
        if mapped_policy in vocab.word2id:
            target_id = vocab.word2id[mapped_policy]
        else:
            try:
                float(mapped_policy)
                target_id = vocab.word2id.get('<C>', -1)
            except (ValueError, TypeError):
                continue
 
        if target_id < 0 or len(input_ids) > max_length or len(input_ids) == 0:
            continue
 
        pairs.append((input_ids, target_id))
 
    return pairs
 
 
def build_training_pairs_from_full_sequence(prefix, root_type, vocab, max_length=30):
    """
    Fallback: 简单 teacher-forcing，按前缀顺序依次预测每个 token。
    """
    pairs = []
 
    # 映射 token 名 + 替换数值系数
    clean_prefix = []
    for item in prefix:
        if isinstance(item, str):
            if item not in GDExpr.word2id and item != '<C>':
                try:
                    float(item)
                    clean_prefix.append('<C>')
                    continue
                except (ValueError, TypeError):
                    pass
            mapped = ND2_TO_NEWDSO_TOKEN.get(item, item)
            clean_prefix.append(mapped)
 
    token_ids = []
    for tok in clean_prefix:
        if tok in vocab.word2id:
            token_ids.append(vocab.word2id[tok])
 
    if len(token_ids) < 2:
        return pairs
 
    root_type_id = vocab.word2id.get(root_type, -1)
    if root_type_id < 0:
        root_type_id = vocab.word2id.get('node', -1)
 
    for i in range(len(token_ids)):
        input_ids = [vocab.sos_id, root_type_id] + token_ids[:i]
        target_id = token_ids[i]
        if len(input_ids) > max_length:
            continue
        pairs.append((input_ids, target_id))
 
    return pairs
 
 
class PretrainDataset(Dataset):
    def __init__(self, equations, vocab, max_length=30, n_repeat=20):
        self.pairs = []
        n_decompose = 0
        n_fallback = 0
 
        for prefix, root_type, name in equations:
            # 优先用 decompose
            try:
                pairs = build_training_pairs_from_equation(
                    prefix, root_type, vocab, max_length)
                if len(pairs) > 0:
                    self.pairs.extend(pairs * n_repeat)
                    n_decompose += len(pairs)
                    continue
            except Exception:
                pass
 
            # Fallback
            try:
                pairs = build_training_pairs_from_full_sequence(
                    prefix, root_type, vocab, max_length)
                self.pairs.extend(pairs * n_repeat)
                n_fallback += len(pairs)
                print(f"[Pretrain] {name} 使用 fallback，生成 {len(pairs)} 个样本")
            except Exception as e:
                print(f"[Pretrain] 跳过 {name}: {e}")
 
        print(f"[Pretrain] 共提取 {len(self.pairs)} 个训练样本 "
              f"(decompose={n_decompose}, fallback={n_fallback})")
 
    def __len__(self):
        return len(self.pairs)
 
    def __getitem__(self, idx):
        input_ids, target_id = self.pairs[idx]
        return torch.LongTensor(input_ids), target_id
 
 
def collate_fn(batch):
    input_ids_list, target_ids = zip(*batch)
    max_len = max(len(x) for x in input_ids_list)
    padded_input = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, seq in enumerate(input_ids_list):
        padded_input[i, :len(seq)] = seq
    target_ids = torch.LongTensor(target_ids)
    lengths = torch.LongTensor([len(x) for x in input_ids_list])
    return padded_input, target_ids, lengths
 
 
def pretrain():
    config = get_default_config()
    config.policy.d_model = 128
    config.policy.nhead = 4
    config.policy.num_layers = 3
    config.policy.dim_feedforward = 256
    config.policy.dropout = 0.1
    config.policy.max_length = 20
 
    seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Pretrain] 设备: {device}")
 
    vocab = Vocabulary(config)
    print(f"[Pretrain] 词汇表大小: {vocab.n_words}, 动作空间: {max(vocab.word2id.values())+1}")
 
    # ★ 硬编码方程
    equations = load_equations_hardcoded()
    print(f"[Pretrain] 加载了 {len(equations)} 个方程")
 
    for prefix, root_type, name in equations:
        try:
            expr_str = GDExpr.prefix2str(prefix)
        except Exception:
            expr_str = ' '.join(str(x) for x in prefix)
        print(f"  [{root_type}] {name}: {expr_str}")
 
    dataset = PretrainDataset(equations, vocab, max_length=config.policy.max_length, n_repeat=30)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
 
    policy = TransformerPolicy(vocab, config).to(device)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Pretrain] TransformerPolicy 参数量: {n_params:,}")
 
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )
 
    n_epochs = 10
    best_loss = float('inf')
 
    for epoch in range(n_epochs):
        policy.train()
        total_loss = 0
        correct = 0
        total = 0
        n_batches = 0
 
        for input_ids, target_ids, lengths in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
 
            all_logits = policy.forward_full(input_ids)
            batch_indices = torch.arange(input_ids.size(0), device=device)
            last_positions = (lengths - 1).clamp(min=0).to(device)
            last_logits = all_logits[batch_indices, last_positions]
 
            loss = F.cross_entropy(last_logits, target_ids, label_smoothing=0.1)
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
 
            total_loss += loss.item()
            correct += (last_logits.argmax(dim=-1) == target_ids).sum().item()
            total += target_ids.size(0)
            n_batches += 1
 
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        acc = correct / max(total, 1)
 
        # 诊断
        if (epoch + 1) % 5 == 0 or epoch == 0:
            policy.eval()
            with torch.no_grad():
                sample_input = input_ids[:1]
                sample_logits = policy.forward_step(sample_input)
                probs = F.softmax(sample_logits, dim=-1)
                top5_vals, top5_ids = probs.topk(5, dim=-1)
                top5_str = ', '.join(
                    f"{vocab.id2word.get(tid.item(), '?')}={prob:.3f}"
                    for tid, prob in zip(top5_ids[0], top5_vals[0])
                )
                target_str = vocab.id2word.get(target_ids[0].item(), '?')
                print(f"  [诊断] 目标={target_str}, Top5预测: {top5_str}")
            policy.train()
 
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = './weights/new_dso_pretrained.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'policy': policy.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'acc': acc,
            }, save_path)
 
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Loss: {avg_loss:.4f} | Acc: {acc:.4f} | "
              f"Best Loss: {best_loss:.4f}")
 
    print(f"\n[Pretrain] 完成！权重保存到 ./weights/new_dso_pretrained.pth")
 
 
if __name__ == '__main__':
    pretrain()
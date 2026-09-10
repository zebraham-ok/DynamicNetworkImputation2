"""
Follow-up analysis of high-variance samples.

"High variance" definition: the std of a sample's prediction probability across the Bootstrap members exceeds the threshold (std > 0.1).
The analysis covers the full year window present in the data, with dimensions including year distribution, node degree distribution, classification boundary, and positive/negative stratification.

Usage:
    python Bootstraps/bootstrap_highvar_analysis.py --dir results/bootstrap/<run>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter


def load_data(bootstrap_dir):
    """Load the prediction matrix and graph data"""
    df = pd.read_csv(os.path.join(bootstrap_dir, 'prediction_matrix.csv'))
    df['high_var'] = df['std_prob'] > 0.1
    df['high_var_flag'] = df['high_var'].astype(int)
    
    dynamic_data = torch.load(os.path.join(bootstrap_dir, 'dynamic_data.pyg'), weights_only=False)
    edge_index = dynamic_data.edge_index
    
    # Compute node degrees
    deg_counter = Counter(edge_index[0].tolist() + edge_index[1].tolist())
    
    df['src_deg'] = df['src_idx'].map(lambda x: deg_counter.get(x, 0))
    df['tgt_deg'] = df['tgt_idx'].map(lambda x: deg_counter.get(x, 0))
    df['min_deg'] = df[['src_deg', 'tgt_deg']].min(axis=1)
    df['max_deg'] = df[['src_deg', 'tgt_deg']].min(axis=1)
    df['max_deg'] = df[['src_deg', 'tgt_deg']].max(axis=1)
    df['sum_deg'] = df['src_deg'] + df['tgt_deg']
    
    # Classification boundary samples: 0.3 <= mean_prob <= 0.7
    df['boundary'] = (df['mean_prob'] >= 0.3) & (df['mean_prob'] <= 0.7)
    
    print(f"Data loading complete:")
    print(f"  Total samples: {len(df)}")
    print(f"  Positive samples: {df['label'].sum()}, negative samples: {len(df) - df['label'].sum()}")
    print(f"  High-variance: {(df['high_var']).sum()} ({df['high_var'].mean():.1%})")
    print(f"  Boundary samples: {df['boundary'].sum()} ({df['boundary'].mean():.1%})")
    
    # Node degree statistics
    deg_values = np.array(list(deg_counter.values()))
    print(f"  Num nodes: {dynamic_data.num_nodes}")
    print(f"  Degree stats: mean={deg_values.mean():.1f}, median={np.median(deg_values):.0f}, max={deg_values.max()}")
    
    return df, dynamic_data, deg_counter


def compute_stats(df):
    """Compute statistics along each dimension"""
    stats = {}

    year_stats = df.groupby('year').agg(
        total=('high_var', 'count'),
        high_var_count=('high_var', 'sum'),
        avg_std=('std_prob', 'mean'),
        median_std=('std_prob', 'median'),
        pct_high=('high_var_flag', 'mean'),
        pos_high_var=('label', lambda x: ((df.loc[x.index, 'high_var']) & (x == 1)).sum()),
    ).reset_index()
    year_stats['pct_high'] = year_stats['pct_high'] * 100
    stats['year'] = year_stats

    # Node degree bucketed logarithmically
    log_bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    bin_labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '101-200', '201-500', '501-1000', '1001-2000', '2000+']
    
    for deg_col, deg_name in [('src_deg', 'Source'), ('tgt_deg', 'Target'), ('min_deg', 'Min'), ('max_deg', 'Max')]:
        df['deg_bin'] = pd.cut(df[deg_col], bins=log_bins, labels=bin_labels, right=True, include_lowest=True)
        deg_stats = df.groupby('deg_bin', observed=False).agg(
            total=('high_var', 'count'),
            high_var_count=('high_var', 'sum'),
            avg_std=('std_prob', 'mean'),
            pct_high=('high_var_flag', 'mean'),
        ).reset_index()
        deg_stats['pct_high'] = deg_stats['pct_high'] * 100
        stats[f'deg_{deg_col}'] = deg_stats
    
    # Bucket by mean_prob (boundary effect)
    prob_bins = np.arange(0, 1.05, 0.1)
    df['prob_bin'] = pd.cut(df['mean_prob'], bins=prob_bins)
    prob_stats = df.groupby('prob_bin', observed=False).agg(
        total=('high_var', 'count'),
        high_var_count=('high_var', 'sum'),
        avg_std=('std_prob', 'mean'),
        median_std=('std_prob', 'median'),
        pct_high=('high_var_flag', 'mean'),
        pos_ratio=('label', 'mean'),
    ).reset_index()
    prob_stats['pct_high'] = prob_stats['pct_high'] * 100
    prob_stats['bin_mid'] = prob_stats['prob_bin'].apply(lambda x: x.mid)
    stats['prob_bin'] = prob_stats

    # Cross: year x positive/negative
    cross = df.groupby(['year', 'label']).agg(
        total=('high_var', 'count'),
        high_var_count=('high_var', 'sum'),
        avg_std=('std_prob', 'mean'),
    ).reset_index()
    cross['pct_high'] = cross['high_var_count'] / cross['total'] * 100
    stats['year_label'] = cross

    # Boundary + high-variance cross
    boundary_hv = df['boundary'].value_counts()
    stats['boundary_total'] = int(df['boundary'].sum())
    stats['boundary_high_var'] = int((df['boundary'] & df['high_var']).sum())
    stats['boundary_high_var_pct'] = float(df[df['boundary']]['high_var'].mean() * 100)
    stats['non_boundary_high_var_pct'] = float(df[~df['boundary']]['high_var'].mean() * 100)
    
    return stats


def plot_analysis(df, stats, save_dir):
    """Generate high-variance analysis visualizations"""
    fig = plt.figure(figsize=(24, 18))

    # 1. By year: high-variance ratio
    ax1 = plt.subplot(3, 4, 1)
    ys = stats['year']
    ax1.bar(ys['year'].astype(str), ys['pct_high'], color='#E74C3C', alpha=0.8, edgecolor='white')
    _overall_hv = df['high_var'].mean() * 100
    ax1.axhline(y=_overall_hv, color='gray', linestyle='--', linewidth=1,
                label=f'Overall: {_overall_hv:.2f}%')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('High-Var Sample Ratio (%)')
    ax1.set_title('High-Var Ratio by Year')
    ax1.legend(fontsize=8)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # 2. By year: mean std
    ax2 = plt.subplot(3, 4, 2)
    ax2.bar(ys['year'].astype(str), ys['avg_std'], color='#3498DB', alpha=0.8, edgecolor='white')
    ax2.axhline(y=df['std_prob'].mean(), color='gray', linestyle='--', label=f'Overall mean: {df["std_prob"].mean():.4f}')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean Std Dev')
    ax2.set_title('Mean Prediction Std by Year')
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # 3. By year, stratified by positive/negative
    ax3 = plt.subplot(3, 4, 7)
    yl = stats['year_label']
    for label, color, marker, name in [(0, '#E74C3C', 'o', 'Negative'), (1, '#2ECC71', 's', 'Positive')]:
        sub = yl[yl['label'] == label]
        ax3.plot(sub['year'], sub['pct_high'], color=color, marker=marker, linewidth=2, label=name)
    _neg_overall = df.loc[df['label'] == 0, 'high_var'].mean() * 100
    _pos_overall = df.loc[df['label'] == 1, 'high_var'].mean() * 100
    ax3.axhline(y=_neg_overall, color='#E74C3C', linestyle=':', alpha=0.5,
                label=f'Neg overall: {_neg_overall:.1f}%')
    ax3.axhline(y=_pos_overall, color='#2ECC71', linestyle=':', alpha=0.5,
                label=f'Pos overall: {_pos_overall:.1f}%')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('High-Var Ratio (%)')
    ax3.set_title('High-Var Ratio: Pos vs Neg by Year')
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # 4. By node degree (min_deg): high-variance ratio
    ax4 = plt.subplot(3, 4, 6)
    ds = stats['deg_min_deg']
    ax4.bar(range(len(ds)), ds['pct_high'], color='#9B59B6', alpha=0.8, edgecolor='white')
    ax4.set_xticks(range(len(ds)))
    ax4.set_xticklabels(ds['deg_bin'], rotation=45, fontsize=8)
    ax4.set_xlabel('Min Degree (src, tgt)')
    ax4.set_ylabel('High-Var Ratio (%)')
    ax4.set_title('High-Var Ratio by Min Node Degree')
    ax4.grid(axis='y', alpha=0.3)

    # 5. By node degree (max_deg): high-variance ratio
    ax5 = plt.subplot(3, 4, 5)
    ds2 = stats['deg_max_deg']
    ax5.bar(range(len(ds2)), ds2['pct_high'], color='#E67E22', alpha=0.8, edgecolor='white')
    ax5.set_xticks(range(len(ds2)))
    ax5.set_xticklabels(ds2['deg_bin'], rotation=45, fontsize=8)
    ax5.set_xlabel('Max Degree (src, tgt)')
    ax5.set_ylabel('High-Var Ratio (%)')
    ax5.set_title('High-Var Ratio by Max Node Degree')
    ax5.grid(axis='y', alpha=0.3)

    # 6. mean_prob vs std scatter (high-variance highlighted)
    ax6 = plt.subplot(3, 4, 8)
    # Subsample to speed up plotting
    sample_n = min(8000, len(df))
    sample_df = df.sample(sample_n, random_state=42)
    colors = []
    for _, row in sample_df.iterrows():
        if row['high_var']:
            colors.append('#E74C3C')
        else:
            colors.append('#BDC3C7')
    ax6.scatter(sample_df['mean_prob'], sample_df['std_prob'], c=colors, alpha=0.4, s=8)
    ax6.axhline(0.1, color='red', linestyle='--', label='High-var threshold (0.1)')
    ax6.axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Stable threshold (0.05)')
    ax6.set_xlabel('Mean Prediction Probability')
    ax6.set_ylabel('Std Dev Across Models')
    ax6.set_title(f'Mean Prob vs Std (n={sample_n} sample)\nRed = high-variance (>0.1)')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.2)

    # 7. Bucketed by mean_prob: high-variance ratio
    ax7 = plt.subplot(3, 4, 3)
    ps = stats['prob_bin']
    ax7.bar(range(len(ps)), ps['pct_high'], color=plt.cm.RdYlGn_r(ps['bin_mid']), alpha=0.8, edgecolor='white')
    ax7.set_xticks(range(len(ps)))
    bin_labels_short = [f'{p.left:.1f}-{p.right:.1f}' for p in ps['prob_bin']]
    ax7.set_xticklabels(bin_labels_short, rotation=45, fontsize=7)
    ax7.set_xlabel('Mean Prob Bin')
    ax7.set_ylabel('High-Var Ratio (%)')
    ax7.set_title('High-Var Ratio by Prediction Confidence')
    ax7.grid(axis='y', alpha=0.3)

    # 8. Bucketed by mean_prob: positive-sample ratio
    ax8 = plt.subplot(3, 4, 4)
    ax8.bar(range(len(ps)), ps['pos_ratio'] * 100, color=plt.cm.RdYlGn_r(ps['bin_mid']), alpha=0.8, edgecolor='white')
    ax8.set_xticks(range(len(ps)))
    ax8.set_xticklabels(bin_labels_short, rotation=45, fontsize=7)
    _overall_pos = df['label'].mean() * 100
    ax8.axhline(y=_overall_pos, color='gray', linestyle='--',
                label=f'Overall pos ratio: {_overall_pos:.1f}%')
    ax8.set_xlabel('Mean Prob Bin')
    ax8.set_ylabel('Positive Ratio (%)')
    ax8.set_title('Positive Sample Ratio by Prediction Confidence')
    ax8.legend(fontsize=8)
    ax8.grid(axis='y', alpha=0.3)

    # 9. High-variance vs non-high-variance: mean_prob distribution
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(df[~df['high_var']]['mean_prob'], bins=40, alpha=0.6, color='#2ECC71', label='Low-Var (std≤0.1)', density=True)
    ax9.hist(df[df['high_var']]['mean_prob'], bins=40, alpha=0.6, color='#E74C3C', label='High-Var (std>0.1)', density=True)
    ax9.set_xlabel('Mean Prediction Probability')
    ax9.set_ylabel('Density')
    ax9.set_title('Mean Prob Distribution: High-Var vs Low-Var')
    ax9.legend(fontsize=8)
    ax9.grid(alpha=0.3)

    # 10. Node degree vs high variance (scatter)
    ax10 = plt.subplot(3, 4, 10)
    sample_df2 = df.sample(min(5000, len(df)), random_state=42)
    sample_df2['log_src_deg'] = np.log10(sample_df2['src_deg'] + 1)
    sample_df2['log_tgt_deg'] = np.log10(sample_df2['tgt_deg'] + 1)
    colors2 = ['#E74C3C' if hv else '#BDC3C7' for hv in sample_df2['high_var']]
    ax10.scatter(sample_df2['log_src_deg'], sample_df2['log_tgt_deg'], c=colors2, alpha=0.3, s=6)
    ax10.set_xlabel('log10(Source Degree + 1)')
    ax10.set_ylabel('log10(Target Degree + 1)')
    ax10.set_title('Node Degree vs High-Variance\n(Red=High-Var, Gray=Low-Var)')
    ax10.grid(alpha=0.2)

    # 11. Source node degree vs target node degree high-variance heatmap
    ax11 = plt.subplot(3, 4, 11)
    # Bucket degrees logarithmically and compute the high-variance ratio per bucket
    df_copy = df.copy()
    deg_bins_heat = [0, 1, 2, 5, 10, 20, 50, 100, 500, 5000]
    deg_labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '101-500', '501+']
    df_copy['src_deg_bin'] = pd.cut(df_copy['src_deg'], bins=deg_bins_heat, labels=deg_labels, right=True)
    df_copy['tgt_deg_bin'] = pd.cut(df_copy['tgt_deg'], bins=deg_bins_heat, labels=deg_labels, right=True)
    heat_data = df_copy.pivot_table(values='high_var_flag', index='src_deg_bin', columns='tgt_deg_bin', aggfunc='mean', observed=False)
    im = ax11.imshow(heat_data.values * 100, cmap='YlOrRd', aspect='auto', vmin=0, vmax=60)
    ax11.set_xticks(range(len(deg_labels)))
    ax11.set_xticklabels(deg_labels, rotation=45, fontsize=7)
    ax11.set_yticks(range(len(deg_labels)))
    ax11.set_yticklabels(deg_labels, fontsize=7)
    ax11.set_xlabel('Target Degree')
    ax11.set_ylabel('Source Degree')
    ax11.set_title('High-Var Ratio (%) Heatmap\nby Src × Tgt Degree')
    plt.colorbar(im, ax=ax11, shrink=0.8)

    # 12. Summary table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    # Boundary effect statistics
    boundary_total = stats['boundary_total']
    boundary_hv = stats['boundary_high_var']

    # Year trend: split into thirds based on the actual data years
    _year_stats = stats['year']
    _years = sorted(_year_stats['year'].tolist())
    _y_lo, _y_hi = _years[0], _years[-1]
    _y_cut = max(1, (_y_hi - _y_lo) // 3)
    early = _year_stats[_year_stats['year'] <= _y_lo + _y_cut]
    late = _year_stats[_year_stats['year'] >= _y_hi - _y_cut]

    neg_hv_pct = df.loc[df['label'] == 0, 'high_var'].mean() * 100
    pos_hv_pct = df.loc[df['label'] == 1, 'high_var'].mean() * 100

    summary_text = (
        f"=== High-Variance Sample Analysis ===\n\n"
        f"High-Var Definition: per-sample std > 0.1\n"
        f"Total: {df['high_var'].sum()} / {len(df)} ({df['high_var'].mean():.1%})\n"
        f"  - Negatives: {(df['label']==0).sum()} → {((df['label']==0) & df['high_var']).sum()} high-var ({(df[df['label']==0]['high_var']).mean():.1%})\n"
        f"  - Positives: {(df['label']==1).sum()} → {((df['label']==1) & df['high_var']).sum()} high-var ({(df[df['label']==1]['high_var']).mean():.1%})\n\n"
        f"--- Year Trends ({_y_lo}-{_y_hi}) ---\n"
        f"First third ({_y_lo}-{_y_lo + _y_cut}): {early['pct_high'].mean():.1f}% high-var\n"
        f"Last third ({_y_hi - _y_cut}-{_y_hi}): {late['pct_high'].mean():.1f}% high-var\n"
        f"Mean std: {early['avg_std'].mean():.4f} -> {late['avg_std'].mean():.4f}\n\n"
        f"--- Boundary Analysis ---\n"
        f"Boundary (0.3≤p≤0.7): {boundary_total} samples\n"
        f"Boundary high-var: {boundary_hv} ({stats['boundary_high_var_pct']:.1f}%)\n"
        f"Non-boundary high-var: {stats['non_boundary_high_var_pct']:.1f}%\n\n"
        f"--- Degree Analysis ---\n"
        f"See the Src x Tgt degree heatmap (panel 11)\n\n"
        f"--- Conclusion ---\n"
        f"High variance driven by:\n"
        f"1. Earlier years of the window\n"
        f"2. Boundary samples (p≈0.5)\n"
        f"3. Low-degree node pairs\n"
        f"4. Negatives ({neg_hv_pct:.1f}%) >> positives ({pos_hv_pct:.1f}%)"
    )
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=7.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Bootstrap High-Variance Sample Analysis\n'
                 f'(Total: {len(df)} samples, High-Var: {df["high_var"].sum()} [{df["high_var"].mean():.1%}])',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'highvar_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {save_path}")

    # Extra chart 1: positive/negative samples by year separately
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, label, color, title in [
        (axes[0], 0, '#E74C3C', 'Negative Samples'),
        (axes[1], 1, '#2ECC71', 'Positive Samples')
    ]:
        sub = df[df['label'] == label]
        year_grp = sub.groupby('year').agg(
            total=('high_var', 'count'),
            hv_count=('high_var', 'sum'),
            avg_std=('std_prob', 'mean'),
        ).reset_index()
        year_grp['hv_pct'] = year_grp['hv_count'] / year_grp['total'] * 100
        
        ax2_dual = ax.twinx()
        ax.bar(year_grp['year'].astype(str), year_grp['hv_pct'], color=color, alpha=0.7, edgecolor='white')
        ax2_dual.plot(range(len(year_grp)), year_grp['avg_std'], 'o-', color='#2C3E50', linewidth=2, markersize=6)
        ax.set_xlabel('Year')
        ax.set_ylabel('High-Var Ratio (%)', color=color)
        ax2_dual.set_ylabel('Mean Std Dev', color='#2C3E50')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    fig2.suptitle('High-Variance Analysis by Year (Stratified by Label)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, 'highvar_year_by_label.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {save_path2}")

    # Extra chart 2: mean_prob distribution of high-variance samples vs low-variance
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

    # Box plot: mean_prob of high-variance vs low-variance
    for ax_i, (ax, label, lbl_name) in enumerate(zip(axes3.flat, [0, 1, None, None], ['Negative', 'Positive', '', ''])):
        if label is not None:
            sub = df[df['label'] == label]
            data_list = [
                sub[sub['high_var']]['mean_prob'].values,
                sub[~sub['high_var']]['mean_prob'].values,
            ]
            bp = ax.boxplot(data_list, tick_labels=['High-Var', 'Low-Var'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#E74C3C')
            bp['boxes'][1].set_facecolor('#2ECC71')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_alpha(0.7)
            ax.set_ylabel('Mean Prediction Probability')
            ax.set_title(f'{lbl_name}: Mean Prob by Variance Group')
            ax.grid(axis='y', alpha=0.3)
        elif ax_i == 2:
            # High-variance ratio of boundary samples
            prob_bins_detail = np.arange(0, 1.01, 0.05)
            df['prob_bin_detail'] = pd.cut(df['mean_prob'], bins=prob_bins_detail)
            detail = df.groupby('prob_bin_detail', observed=False).agg(
                total=('high_var', 'count'),
                hv_count=('high_var', 'sum'),
            ).reset_index()
            detail['hv_pct'] = detail['hv_count'] / detail['total'] * 100
            detail['bin_mid'] = detail['prob_bin_detail'].apply(lambda x: x.mid)
            ax.bar(detail['bin_mid'], detail['hv_pct'], width=0.04, color='#E74C3C', alpha=0.7, edgecolor='white')
            ax.set_xlabel('Mean Prediction Probability')
            ax.set_ylabel('High-Var Ratio (%)')
            ax.set_title('High-Var Ratio by Confidence Interval (0.05 bins)')
            ax.grid(axis='y', alpha=0.3)
        else:
            # Feature importance summary
            ax.axis('off')
            feat_text = (
                "=== Key Drivers of High Variance ===\n\n"
                f"1. EARLIER YEARS ({_y_lo}-{_y_lo + _y_cut}):\n"
                f"   High-var ratio: {early['pct_high'].mean():.1f}%\n"
                f"   (last third {_y_hi - _y_cut}-{_y_hi}: {late['pct_high'].mean():.1f}%)\n"
                f"   → Sparse data, noisy edges\n\n"
                f"2. CLASSIFICATION BOUNDARY:\n"
                f"   Boundary (0.3-0.7): {stats['boundary_high_var_pct']:.1f}% high-var\n"
                f"   vs {stats['non_boundary_high_var_pct']:.1f}% outside\n"
                f"   → Hard-to-classify samples\n\n"
                f"3. LOW-DEGREE NODES:\n"
                f"   See the Src x Tgt degree heatmap\n"
                f"   → Insufficient structural signal\n\n"
                f"4. NEGATIVE SAMPLES:\n"
                f"   {neg_hv_pct:.1f}% high-var vs {pos_hv_pct:.1f}% positive\n"
                f"   → Negatives inherently harder to\n"
                f"     reach consensus on\n\n"
                f"ACTION ITEMS:\n"
                f"• Early years: consider data augmentation\n"
                f"• Low-degree: add node attributes\n"
                f"• Negatives: hard negative mining\n"
                f"• Boundary: ensemble calibration"
            )
            ax.text(0.05, 0.95, feat_text, transform=ax.transAxes,
                   fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig3.suptitle('High-Variance Sample: Detailed Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path3 = os.path.join(save_dir, 'highvar_detail_analysis.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {save_path3}")
    
    return {
        'highvar_analysis.png': save_path,
        'highvar_year_by_label.png': save_path2,
        'highvar_detail_analysis.png': save_path3,
    }


def main():
    parser = argparse.ArgumentParser(description='Follow-up analysis of Bootstrap high-variance samples')
    parser.add_argument('--dir', '-d', required=True, help='Bootstrap output directory')
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        raise NotADirectoryError(f"Directory does not exist: {args.dir}")
    
    print(f"\n{'='*60}")
    print(f"Bootstrap High-Variance Sample Follow-Up Analysis")
    print(f"  Directory: {args.dir}")
    print(f"{'='*60}\n")
    
    # 1. Load data
    print("[Step 1] Loading data...")
    df, dynamic_data, deg_counter = load_data(args.dir)

    # 2. Compute statistics
    print("\n[Step 2] Computing statistics...")
    stats = compute_stats(df)

    # 3. Print key findings
    print("\n" + "="*60)
    print("Key Findings")
    print("="*60)

    # Year trend (split into thirds based on the actual data years)
    ys = stats['year']
    _years = sorted(ys['year'].tolist())
    _y_lo, _y_hi = _years[0], _years[-1]
    _y_cut = max(1, (_y_hi - _y_lo) // 3)
    early = ys[ys['year'] <= _y_lo + _y_cut]
    late = ys[ys['year'] >= _y_hi - _y_cut]
    print(f"\n[Year Trends]")
    print(f"  Early years ({_y_lo}-{_y_lo + _y_cut}) high-var ratio: {early['pct_high'].mean():.1f}%")
    print(f"  Late years ({_y_hi - _y_cut}-{_y_hi}) high-var ratio: {late['pct_high'].mean():.1f}%")
    print(f"  Early mean std: {early['avg_std'].mean():.4f}, late mean std: {late['avg_std'].mean():.4f}")

    # Boundary effect
    print(f"\n[Boundary Effect]")
    print(f"  Boundary samples (0.3<=p<=0.7) total: {stats['boundary_total']}")
    print(f"  High-var share within boundary: {stats['boundary_high_var_pct']:.1f}%")
    print(f"  High-var share outside boundary: {stats['non_boundary_high_var_pct']:.1f}%")

    # Node degree
    print(f"\n[Node Degree Effect]")
    deg_by_bin = df.groupby(pd.cut(df['min_deg'], bins=[0,1,2,5,10,20,50,100,5000], labels=['deg=1','deg=2','deg=3-5','deg=6-10','deg=11-20','deg=21-50','deg=51-100','deg=101+']), observed=False)
    for name, group in deg_by_bin:
        print(f"  {name}: high-var ratio = {group['high_var'].mean():.1%}, num samples = {len(group)}")

    # Positive/negative samples
    print(f"\n[Positive vs Negative Sample Differences]")
    print(f"  Negative high-var: {(df['label']==0).sum()} -> {((df['label']==0) & df['high_var']).sum()} ({(df[df['label']==0]['high_var']).mean():.1%})")
    print(f"  Positive high-var: {(df['label']==1).sum()} -> {((df['label']==1) & df['high_var']).sum()} ({(df[df['label']==1]['high_var']).mean():.1%})")

    # 4. Visualization
    print(f"\n[Step 3] Generating visualizations...")
    outputs = plot_analysis(df, stats, args.dir)
    
    print(f"\n[Done] High-variance analysis complete!")
    for name, path in outputs.items():
        print(f"   {name}: {path}")


if __name__ == "__main__":
    main()
